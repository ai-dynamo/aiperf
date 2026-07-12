# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config v2 -> v2 runner -> direct online Graph-IR product proof."""

from __future__ import annotations

import copy
import os
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.orchestrator.runner_installation import RunnerInstallation
from aiperf.orchestrator.rust_executor import RustSubprocessExecutor


def _graph_rows() -> list[dict[str, Any]]:
    return [
        {
            "session_id": "root",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "root-0"}],
                    "forks": [{"child": "fork", "background": True}],
                    "spawns": [{"children": ["spawn"], "join_at": 1}],
                    "max_tokens": 1,
                },
                {
                    "messages": [{"role": "user", "content": "root-1"}],
                    "max_tokens": 1,
                },
            ],
        },
        {
            "session_id": "fork",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "fork-0"}],
                    "max_tokens": 1,
                }
            ],
        },
        {
            "session_id": "spawn",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "spawn-0"}],
                    "max_tokens": 1,
                }
            ],
        },
    ]


class _GraphChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    bodies: list[dict[str, Any]] = []
    bodies_lock = threading.Lock()

    def setup(self) -> None:
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = orjson.loads(self.rfile.read(length))
        assert isinstance(body, dict)
        with self.bodies_lock:
            self.bodies.append(body)

        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        messages = body.get("messages")
        assert isinstance(messages, list)
        last_user = next(
            message["content"]
            for message in reversed(messages)
            if message.get("role") == "user"
        )
        response = b"".join(
            [
                b"data: "
                + orjson.dumps(
                    {"choices": [{"delta": {"content": f"answer-{last_user}"}}]}
                )
                + b"\n\n",
                b'data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":1}}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        pass


def _runner_binary() -> Path:
    default = Path(__file__).resolve().parents[2] / "target/debug/aiperf-runner"
    return Path(os.environ.get("AIPERF_RUNNER_BIN", default))


@pytest.fixture(scope="module")
def online_graph_installation() -> RunnerInstallation:
    installation = RunnerInstallation.resolve(_runner_binary())
    assert installation.supports_pair("online_http", "graph")
    return installation


def _graph_run(
    artifact_dir: Path,
    endpoint_url: str,
    *,
    phases: list[dict[str, Any]] | None = None,
    benchmark_id: str = "python-v2-online-graph",
) -> BenchmarkRun:
    config = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["mock-model"],
                "endpoint": {
                    "urls": [endpoint_url],
                    "type": "chat",
                    "streaming": True,
                    "use_server_token_count": True,
                },
                "dataset": {
                    "type": "file",
                    "format": "dag_jsonl",
                    "sampling": "sequential",
                    "records": _graph_rows(),
                },
                "phases": phases
                or [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 4,
                        "concurrency": 2,
                    }
                ],
                "tokenizer": {"name": "builtin"},
                "runtime": {"workers": 2, "ui": "none"},
                "gpu_telemetry": {"enabled": False},
                "server_metrics": {"enabled": False},
                "network_latency": {"enabled": False},
                "artifacts": {
                    "dir": str(artifact_dir),
                    "records": False,
                    "raw": False,
                    "trace": False,
                    "export_outputs_json": False,
                },
            }
        }
    )
    return BenchmarkRun(
        benchmark_id=benchmark_id,
        cfg=config.benchmark,
        artifact_dir=artifact_dir,
        label="online-graph",
        random_seed=19,
    )


def _execute_v2(
    monkeypatch: pytest.MonkeyPatch,
    installation: RunnerInstallation,
    run: BenchmarkRun,
) -> tuple[dict[str, Any], subprocess.CompletedProcess[bytes], Any]:
    original_execute = RunnerInstallation.execute
    captured: list[tuple[dict[str, Any], subprocess.CompletedProcess[bytes]]] = []

    def recording_execute(
        selected: RunnerInstallation, request: dict[str, Any]
    ) -> subprocess.CompletedProcess[bytes]:
        completed = original_execute(selected, request)
        captured.append((copy.deepcopy(request), completed))
        return completed

    monkeypatch.setattr(RunnerInstallation, "execute", recording_execute)
    result = RustSubprocessExecutor(
        run.artifact_dir,
        installation=installation,
    ).execute_sync(run)

    assert len(captured) == 1
    request, completed = captured[0]
    return request, completed, result


def _message_contents(body: dict[str, Any]) -> list[str]:
    return [
        content
        for message in body["messages"]
        if isinstance((content := message.get("content")), str)
    ]


def test_python_config_v2_reaches_online_graph_adapter_without_dual_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    online_graph_installation: RunnerInstallation,
) -> None:
    with _GraphChatHandler.bodies_lock:
        _GraphChatHandler.bodies.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GraphChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        artifact_dir = tmp_path / "online-graph"
        endpoint_url = (
            f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        )
        run = _graph_run(artifact_dir, endpoint_url)

        request, completed, result = _execute_v2(
            monkeypatch,
            online_graph_installation,
            run,
        )

        assert request["protocol_version"] == 2
        assert request["operation"] == "execute"
        assert (
            request["expected_distribution_id"]
            == online_graph_installation.distribution_id
        )
        assert request["run"]["backend"] == {"type": "online_http", "config": {}}
        assert request["run"]["workload"]["type"] == "graph"
        dataset = request["run"]["workload"]["config"]["dataset"]
        assert dataset["format"] == "dag_jsonl"
        assert dataset["records"] == _graph_rows()
        assert "conversation" not in dataset
        assert "graph_ir" not in dataset

        terminal = orjson.loads(completed.stdout)
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
        assert terminal["success"] is True
        assert terminal["distribution_id"] == online_graph_installation.distribution_id
        assert terminal["provenance"] == {
            "backend": "online_http",
            "workload": "graph",
        }
        assert result.success, result.error
        assert result.summary_metrics["request_count"].avg == 4.0

        native_path = artifact_dir / "native-v2.json"
        assert Path(terminal["report_path"]) == native_path
        native = orjson.loads(native_path.read_bytes())
        assert native["run"]["mode"] == "graph"
        assert native["run"]["backend"] == "online_http"
        assert native["run"]["workload"] == "graph"
        assert native["run"]["graph"] == {
            "input_format": "dag_jsonl",
            "root_count": 1,
            "node_count": 4,
            "worker_count": 2,
            "phase_count": 1,
        }

        with _GraphChatHandler.bodies_lock:
            bodies = copy.deepcopy(_GraphChatHandler.bodies)
        assert len(bodies) == 4
        histories = {
            contents[-1]: contents
            for body in bodies
            if (contents := _message_contents(body))
        }
        assert histories == {
            "root-0": ["root-0"],
            "fork-0": ["root-0", "answer-root-0", "fork-0"],
            "spawn-0": ["spawn-0"],
            "root-1": ["root-0", "answer-root-0", "root-1"],
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_python_config_v2_graph_uses_shared_phase_ramp_adaptive_and_session_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    online_graph_installation: RunnerInstallation,
) -> None:
    with _GraphChatHandler.bodies_lock:
        _GraphChatHandler.bodies.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GraphChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        artifact_dir = tmp_path / "online-graph-phases"
        endpoint_url = (
            f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        )
        run = _graph_run(
            artifact_dir,
            endpoint_url,
            benchmark_id="python-v2-online-graph-phases",
            phases=[
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "sessions": 1,
                    "concurrency": 2,
                    "concurrency_ramp": {
                        "duration": 0.01,
                        "strategy": "linear",
                    },
                },
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "sessions": 1,
                    "duration": 0.05,
                    "concurrency": 2,
                    "seamless": True,
                    "prefill_concurrency": 2,
                    "grace_period": 0.01,
                    "adaptive_scale": {
                        "enabled": True,
                        "control": {
                            "variable": "prefill_concurrency",
                            "min": 1,
                            "max": 2,
                        },
                        "assessment_period": 1,
                        "min_completed_requests": 1,
                        "sustain_duration": 1,
                        "strategy": {
                            "type": "ramp_until_fail",
                            "step_policy": "fixed_percent_step",
                            "step_percent": 100,
                        },
                    },
                    "sla": {
                        "request_latency": {"p95": {"le": 1000}},
                    },
                },
            ],
        )

        request, completed, result = _execute_v2(
            monkeypatch,
            online_graph_installation,
            run,
        )

        assert request["protocol_version"] == 2
        assert request["run"]["workload"]["type"] == "graph"
        projected_phases = request["run"]["workload"]["config"]["phases"]
        assert projected_phases[0]["seamless"] is False
        assert projected_phases[1]["seamless"] is True
        assert projected_phases[0]["concurrency_ramp"] == {
            "duration": 0.01,
            "strategy": "linear",
        }
        assert projected_phases[1]["adaptive_scale"]["control_variable"] == (
            "prefill_concurrency"
        )

        terminal = orjson.loads(completed.stdout)
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
        assert terminal["success"] is True
        assert result.success, result.error
        assert result.summary_metrics["request_count"].avg == 4.0
        assert (artifact_dir / "adaptive_scale_events.jsonl").is_file()
        assert (artifact_dir / "adaptive_scale_summary.json").is_file()

        with _GraphChatHandler.bodies_lock:
            bodies = copy.deepcopy(_GraphChatHandler.bodies)
        assert len(bodies) == 8
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
