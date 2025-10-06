# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import orjson


@dataclass
class AIPerfSubprocessResult:
    """AIPerf subprocess result."""

    exit_code: int
    stdout: str
    stderr: str
    output_dir: Path


@dataclass
class FakeAIServer:
    """FakeAI server info."""

    host: str
    port: int
    url: str
    process: object


class AIPerfResults:
    """Simple wrapper for AIPerf results."""

    def __init__(self, result: AIPerfSubprocessResult) -> None:
        self.artifacts_dir = result.output_dir
        self.exit_code = result.exit_code

        # Load all artifacts upfront
        json_file = next(self.artifacts_dir.glob("**/*aiperf.json"), None)
        self._data = orjson.loads(json_file.read_text()) if json_file else {}

        csv_file = next(self.artifacts_dir.glob("**/*aiperf.csv"), None)
        self._csv = csv_file.read_text() if csv_file else ""

        inputs_file = next(self.artifacts_dir.glob("**/inputs.json"), None)
        self._inputs = orjson.loads(inputs_file.read_text()) if inputs_file else {}

    @property
    def metrics(self) -> dict:
        """Get all metrics."""
        return self._data.get("records", {})

    @property
    def request_count(self) -> int:
        """Get number of completed requests."""
        return int(self.metrics.get("request_count", {}).get("avg", 0))

    @property
    def has_streaming_metrics(self) -> bool:
        """Check if streaming metrics exist."""
        return "ttft" in self.metrics and "inter_token_latency" in self.metrics

    @property
    def csv_content(self) -> str:
        """Get CSV content."""
        return self._csv

    @property
    def has_input_images(self) -> bool:
        """Check if inputs contain images."""
        return "image_url" in orjson.dumps(self._inputs).decode("utf-8")

    @property
    def has_input_audio(self) -> bool:
        """Check if inputs contain audio."""
        return "input_audio" in orjson.dumps(self._inputs).decode("utf-8")


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
        """Run aiperf command and return results."""
        args = self._parse_command(command)
        result = await self._runner(args, timeout)

        if assert_success:
            assert result.exit_code == 0, (
                f"AIPerf process failed with exit code {result.exit_code}"
            )

        return AIPerfResults(result)

    @staticmethod
    def _parse_command(cmd: str) -> list[str]:
        """Parse command string into args."""
        cmd = cmd.strip().replace("\\\n", " ")
        args = shlex.split(cmd)
        return args[1:] if args and args[0] == "aiperf" else args
