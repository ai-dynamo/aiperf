# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Python outer loops driving one fresh Rust process per benchmark run."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.config import AIPerfConfig
from aiperf.config.loader.plan import build_benchmark_plan
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.rust_executor import RustSubprocessExecutor

_SSE = b"".join(
    [
        b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}]}\n\n',
        b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":1}}\n\n',
        b"data: [DONE]\n\n",
    ]
)


@dataclass
class _ServerState:
    bodies: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, body: dict[str, Any]) -> None:
        with self.lock:
            self.bodies.append(body)

    def count(self) -> int:
        with self.lock:
            return len(self.bodies)


class _ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = orjson.loads(self.rfile.read(length))
        self.server.state.append(body)  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(_SSE)))
        self.end_headers()
        self.wfile.write(_SSE)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def native_server():
    state = _ServerState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def runner_binary() -> Path:
    default = Path(__file__).parents[2] / "target/debug/aiperf-runner"
    binary = Path(os.environ.get("AIPERF_RUNNER_BIN", default))
    assert binary.is_file(), f"build aiperf-runner before this integration test: {binary}"
    return binary


def _config(url: str, artifact_dir: Path, *, sweep: dict[str, Any]) -> AIPerfConfig:
    return AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["mock-model"],
                "endpoint": {
                    "urls": [url],
                    "type": "chat",
                    "streaming": True,
                    "use_server_token_count": True,
                },
                "datasets": [
                    {
                        "name": "profiling",
                        "type": "synthetic",
                        "entries": 8,
                        "prompts": {"isl": 8, "osl": 1},
                    }
                ],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 2,
                        "concurrency": 1,
                    }
                ],
                "artifacts": {"dir": str(artifact_dir), "records": ["jsonl"]},
                "gpu_telemetry": {"enabled": False},
                "server_metrics": {"enabled": False},
                "runtime": {"ui": "none"},
            },
            "random_seed": 41,
            "sweep": sweep,
        }
    )


async def _execute(config: AIPerfConfig, root: Path, binary: Path):
    plan = build_benchmark_plan(config)
    executor = RustSubprocessExecutor(root, binary=binary)
    results = await MultiRunOrchestrator(root).execute(plan, executor)
    assert results and all(result.success for result in results), [
        result.error for result in results
    ]
    for result in results:
        assert result.artifacts_path is not None
        assert (result.artifacts_path / "native-v2.json").is_file()
        assert (result.artifacts_path / "profile_export.jsonl").is_file()
    return plan, results


@pytest.mark.asyncio
async def test_grid_cartesian_product_runs_all_coordinates_in_rust(
    tmp_path: Path, native_server, runner_binary: Path
) -> None:
    url, state = native_server
    before = state.count()
    config = _config(
        url,
        tmp_path,
        sweep={
            "type": "grid",
            "parameters": {
                "phases.profiling.concurrency": [1, 2],
                "datasets.profiling.prompts.isl": [8, 12],
            },
        },
    )

    plan, results = await _execute(config, tmp_path, runner_binary)

    assert len(plan.variations) == len(results) == 4
    assert {
        tuple(sorted(result.variation_values.items())) for result in results
    } == {
        (
            ("datasets.profiling.prompts.isl", isl),
            ("phases.profiling.concurrency", concurrency),
        )
        for concurrency in (1, 2)
        for isl in (8, 12)
    }
    assert state.count() - before == 8


@pytest.mark.asyncio
async def test_zip_and_scenarios_preserve_authored_pairing(
    tmp_path: Path, native_server, runner_binary: Path
) -> None:
    url, state = native_server
    before = state.count()
    zip_root = tmp_path / "zip"
    zip_config = _config(
        url,
        zip_root,
        sweep={
            "type": "zip",
            "parameters": {
                "phases.profiling.concurrency": [1, 2],
                "phases.profiling.requests": [2, 3],
            },
        },
    )
    _, zip_results = await _execute(zip_config, zip_root, runner_binary)
    assert [
        (
            result.variation_values["phases.profiling.concurrency"],
            result.variation_values["phases.profiling.requests"],
        )
        for result in zip_results
    ] == [(1, 2), (2, 3)]

    scenario_root = tmp_path / "scenarios"
    scenario_config = _config(
        url,
        scenario_root,
        sweep={
            "type": "scenarios",
            "runs": [
                {
                    "name": "single",
                    "benchmark": {
                        "phases": [
                            {"name": "profiling", "concurrency": 1, "requests": 2}
                        ]
                    },
                },
                {
                    "name": "paired",
                    "benchmark": {
                        "phases": [
                            {"name": "profiling", "concurrency": 2, "requests": 3}
                        ]
                    },
                },
            ],
        },
    )
    plan, scenario_results = await _execute(
        scenario_config, scenario_root, runner_binary
    )
    assert [variation.label for variation in plan.variations] == ["single", "paired"]
    assert [result.variation_label for result in scenario_results] == [
        "single",
        "paired",
    ]
    assert state.count() - before == 10


@pytest.mark.parametrize("sweep_type", ["sobol", "latin_hypercube"])
@pytest.mark.asyncio
async def test_qmc_design_coordinates_execute_in_rust(
    tmp_path: Path,
    native_server,
    runner_binary: Path,
    sweep_type: str,
) -> None:
    url, state = native_server
    before = state.count()
    config = _config(
        url,
        tmp_path,
        sweep={
            "type": sweep_type,
            "samples": 2,
            "seed": 17,
            "dimensions": [
                {
                    "path": "phases.profiling.concurrency",
                    "lo": 1,
                    "hi": 3,
                    "kind": "int",
                },
                {"path": "phases.profiling.requests", "choices": [2, 3]},
            ],
        },
    )

    plan, results = await _execute(config, tmp_path, runner_binary)

    assert len(results) == 2
    design = orjson.loads(
        (tmp_path / "sweep_aggregate" / "sampling_design.json").read_bytes()
    )
    assert design["samples_mapped"] == [
        [
            result.variation_values["phases.profiling.concurrency"],
            result.variation_values["phases.profiling.requests"],
        ]
        for result in results
    ]
    assert state.count() - before == sum(
        result.variation_values["phases.profiling.requests"] for result in results
    )
