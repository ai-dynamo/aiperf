# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Python outer loops driving one fresh Rust process per benchmark run."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.config import AIPerfConfig
from aiperf.config.loader.plan import build_benchmark_plan
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.native_execution import NativeExecutor

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
    intervals: list[list[float | None]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def begin(self, body: dict[str, Any]) -> int:
        with self.lock:
            self.bodies.append(body)
            self.intervals.append([time.monotonic(), None])
            return len(self.intervals) - 1

    def finish(self, index: int) -> None:
        with self.lock:
            self.intervals[index][1] = time.monotonic()

    def count(self) -> int:
        with self.lock:
            return len(self.bodies)

    def completed_intervals_since(self, index: int) -> list[tuple[float, float]]:
        with self.lock:
            return [
                (start, end)
                for start, end in self.intervals[index:]
                if start is not None and end is not None
            ]


class _ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = orjson.loads(self.rfile.read(length))
        observation = self.server.state.begin(body)  # type: ignore[attr-defined]
        time.sleep(0.01)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(_SSE)))
        self.end_headers()
        self.wfile.write(_SSE)
        self.server.state.finish(observation)  # type: ignore[attr-defined]

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


def _config(
    url: str,
    artifact_dir: Path,
    *,
    sweep: dict[str, Any] | None = None,
    multi_run: dict[str, Any] | None = None,
) -> AIPerfConfig:
    envelope: dict[str, Any] = {
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
        }
    if sweep is not None:
        envelope["sweep"] = sweep
    if multi_run is not None:
        envelope["multi_run"] = multi_run
    return AIPerfConfig.model_validate(envelope)


async def _execute(config: AIPerfConfig, root: Path, binary: Path):
    plan = build_benchmark_plan(config)
    executor = NativeExecutor(root, binary=binary)
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


@pytest.mark.parametrize("iteration_order", ["repeated", "independent"])
@pytest.mark.asyncio
async def test_trials_use_both_iteration_orders_and_canonical_artifact_trees(
    tmp_path: Path,
    native_server,
    runner_binary: Path,
    iteration_order: str,
) -> None:
    url, state = native_server
    before = state.count()
    config = _config(
        url,
        tmp_path,
        sweep={
            "type": "grid",
            "iteration_order": iteration_order,
            "parameters": {"phases.profiling.concurrency": [1, 2]},
        },
        multi_run={"num_runs": 2, "cooldown_seconds": 0},
    )

    _, results = await _execute(config, tmp_path, runner_binary)

    assert len(results) == 4
    assert sorted(result.trial_index for result in results) == [0, 0, 1, 1]
    relative_paths = [result.artifacts_path.relative_to(tmp_path) for result in results]
    if iteration_order == "repeated":
        assert all(path.parts[0] == "profile_runs" for path in relative_paths)
        assert {path.parts[1] for path in relative_paths} == {
            "trial_0001",
            "trial_0002",
        }
    else:
        assert all(
            len(path.parts) == 3 and path.parts[1] == "profile_runs"
            for path in relative_paths
        )
        assert {path.parts[2] for path in relative_paths} == {
            "trial_0001",
            "trial_0002",
        }
    assert state.count() - before == 8


@pytest.mark.parametrize(
    ("mode", "min_runs", "expected_runs"),
    [("cv", 2, 2), ("ci_width", 2, 2), ("distribution", 3, 3)],
)
@pytest.mark.asyncio
async def test_native_samples_drive_all_convergence_modes(
    tmp_path: Path,
    native_server,
    runner_binary: Path,
    mode: str,
    min_runs: int,
    expected_runs: int,
) -> None:
    url, state = native_server
    before = state.count()
    config = _config(
        url,
        tmp_path,
        multi_run={
            "num_runs": 4,
            "cooldown_seconds": 0,
            "convergence": {
                "metric": "output_sequence_length",
                "stat": "avg",
                "mode": mode,
                "threshold": 0.1,
                "min_runs": min_runs,
            },
        },
    )
    config.benchmark.phases[0].requests = 3

    plan, results = await _execute(config, tmp_path, runner_binary)

    assert plan.export_level == "records"
    assert plan.export_jsonl_file == "profile_export.jsonl"
    assert len(results) == expected_runs
    assert state.count() - before == 3 * expected_runs
    for result in results:
        rows = (result.artifacts_path / "profile_export.jsonl").read_text().splitlines()
        assert len(rows) == 3
        assert {
            orjson.loads(row)["metrics"]["output_sequence_length"]["value"]
            for row in rows
        } == {1.0}


def _peak_overlap(intervals: list[tuple[float, float]]) -> int:
    events = [(start, 1) for start, _ in intervals] + [
        (end, -1) for _, end in intervals
    ]
    active = peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], -event[1])):
        active += delta
        peak = max(peak, active)
    return peak


@pytest.mark.asyncio
async def test_two_parameter_adaptive_search_changes_real_rust_load(
    tmp_path: Path, native_server, runner_binary: Path
) -> None:
    from aiperf.cli_runner._strategy import _build_search_planner

    url, state = native_server
    before = state.count()
    config = _config(
        url,
        tmp_path,
        sweep={
            "type": "adaptive_search",
            "planner": "optuna",
            "optuna_sampler": "tpe",
            "search_space": [
                {
                    "path": "phases.profiling.concurrency",
                    "lo": 1,
                    "hi": 3,
                    "kind": "int",
                },
                {
                    "path": "phases.profiling.requests",
                    "lo": 3,
                    "hi": 4,
                    "kind": "int",
                },
            ],
            "objectives": [
                {
                    "metric": "output_token_throughput",
                    "stat": "avg",
                    "direction": "maximize",
                }
            ],
            "max_iterations": 3,
            "n_initial_points": 1,
            "random_seed": 29,
        },
    )
    plan = build_benchmark_plan(config)
    planner = _build_search_planner(plan)
    assert planner is not None
    executor = NativeExecutor(tmp_path, binary=runner_binary)

    results = await MultiRunOrchestrator(tmp_path).execute(
        plan, executor, search_planner=planner
    )

    assert len(results) == 3
    assert all(result.success for result in results)
    assert [result.variation_label for result in results] == [
        "search_iter_0000",
        "search_iter_0001",
        "search_iter_0002",
    ]
    expected_requests = [
        int(result.variation_values["phases.profiling.requests"])
        for result in results
    ]
    expected_concurrency = [
        int(result.variation_values["phases.profiling.concurrency"])
        for result in results
    ]
    assert [int(result.summary_metrics["request_count"].avg) for result in results] == (
        expected_requests
    )
    assert state.count() - before == sum(expected_requests)

    intervals = state.completed_intervals_since(before)
    offset = 0
    observed_peaks: list[int] = []
    for request_count in expected_requests:
        run_intervals = intervals[offset : offset + request_count]
        observed_peaks.append(_peak_overlap(run_intervals))
        offset += request_count
    assert observed_peaks == expected_concurrency

    history = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert len(history["iterations"]) == 3
    assert [entry["variation_values"] for entry in history["iterations"]] == [
        result.variation_values for result in results
    ]
