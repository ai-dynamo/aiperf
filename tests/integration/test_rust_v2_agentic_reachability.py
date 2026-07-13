# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config v2 -> v2 runner -> direct canonical agentic adapter proof."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.orchestrator import rust_executor
from aiperf.orchestrator.runner_installation import RunnerInstallation
from aiperf.orchestrator.rust_executor import RustSubprocessExecutor

pytestmark = pytest.mark.skip(
    reason="product wire no longer projects this mode; modules remain linked for later deletion"
)

_WORKER = r"""
import json
import os
import sys

episode = {
    "episode_id": "episode-1",
    "task": "fixture.agentic",
    "source": "fixture/agentic",
}
events = []
terminal = None

for line in sys.stdin:
    request = json.loads(line)
    operation = request["op"]
    if operation == "hello":
        result = {
            "protocol": 1,
            "worker_version": "python-v2-agentic-fixture",
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": {"fixture-harness": "1.0.0"},
            "worker_source_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "container_digest": None,
            "capabilities": [
                "load",
                "next_problems",
                "grade_batch",
                "shutdown",
                "agentic",
                "agentic_inference_gateway",
            ],
        }
    elif operation == "load_agentic":
        assert request["dataset"] == "fixture/agentic@locked"
        assert request["model"] == "primary-model"
        assert request["config"]["task_concurrency"] == 1
        assert request["config"]["inference_gateway"]["base_url"].startswith(
            "http://127.0.0.1:"
        )
        expected_absent = os.environ["FIXTURE_ARTIFACT_TARGET"]
        assert not os.path.exists(expected_absent), expected_absent
        result = {
            "harness": "fixture-canonical",
            "harness_version": "1.0.0",
            "harness_source_sha256": "c" * 64,
            "dataset": {
                "provider": "fixture",
                "benchmark": "fixture/agentic",
                "repository": "fixture/agentic",
                "revision": "d" * 64,
                "evaluation_splits": ["tasks"],
            },
            "agent": "fixture-agent",
            "agent_version": "1.0.0",
            "environment": "fixture",
            "verifier": "fixture verifier",
            "episode_count": 1,
            "primary_reward": "reward",
        }
    elif operation == "next_episodes":
        items = [episode] if request["offset"] == 0 else []
        result = {"items": items, "next_offset": 1, "done": True}
    elif operation == "start_episodes":
        assert request["episode_ids"] == ["episode-1"]
        events.append(
            {
                "kind": "model_call",
                "call": {
                    "episode_id": "episode-1",
                    "call_id": "episode-1:call:0",
                    "turn_index": 0,
                    "prompt": "Answer the canonical fixture",
                    "messages": [
                        {"role": "user", "content": "Answer the canonical fixture"}
                    ],
                    "generation": {
                        "max_tokens": 8,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "stop": [],
                    },
                },
            }
        )
        result = {"started": request["episode_ids"]}
    elif operation == "poll_agentic":
        page = events[: request["limit"]]
        del events[: len(page)]
        result = {"events": page}
    elif operation == "submit_model_results":
        item = request["items"][0]
        assert item["episode_id"] == "episode-1"
        assert item["call_id"] == "episode-1:call:0"
        assert item["status"] == "completed"
        assert item["response"] == "fixture-answer"
        terminal = {
            "episode_id": "episode-1",
            "task": "fixture.agentic",
            "outcome": "completed",
            "rewards": {"reward": 1.0},
            "primary_reward": "reward",
            "duration_seconds": 0.1,
            "model_calls": 1,
            "prompt_tokens": item.get("prompt_tokens"),
            "completion_tokens": item.get("completion_tokens"),
            "artifact_path": "episodes/episode-1",
        }
        if item.get("cached_tokens") is not None:
            terminal["cached_tokens"] = item["cached_tokens"]
        events.append({"kind": "episode_completed", "result": terminal})
        result = {"accepted": [item["call_id"]]}
    elif operation == "cancel_episodes":
        result = {"cancelled": request["episode_ids"]}
    elif operation == "finish_agentic":
        result = {"items": [terminal]}
    elif operation == "shutdown":
        result = {"shutdown": True}
    else:
        raise RuntimeError(operation)
    print(json.dumps({"id": request["id"], "ok": True, "result": result}), flush=True)
    if operation == "shutdown":
        break
"""


class _AgenticChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    bodies: list[dict[str, Any]] = []
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = orjson.loads(self.rfile.read(length))
        assert isinstance(body, dict)
        with self.lock:
            self.bodies.append(body)
        response = b"".join(
            [
                b'data: {"id":"fixture-response","choices":[{"delta":{"content":"fixture-answer"},"finish_reason":null}]}\n\n',
                b'data: {"id":"fixture-response","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
                b'data: {"id":"fixture-response","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n',
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
def agentic_installation() -> RunnerInstallation:
    installation = RunnerInstallation.resolve(_runner_binary())
    return installation


def _install_worker(root: Path) -> None:
    package = root / "fixture_agentic_v2"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "worker.py").write_text(_WORKER, encoding="utf-8")


def _run(artifact_dir: Path, endpoint_url: str, worker_root: Path) -> BenchmarkRun:
    config = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["primary-model"],
                "endpoint": {
                    "urls": [endpoint_url],
                    "type": "chat",
                    "streaming": True,
                    "use_server_token_count": True,
                },
                "dataset": {
                    "type": "synthetic",
                    "entries": 1,
                    "prompts": {"isl": 4, "osl": 1},
                },
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                    }
                ],
                "tokenizer": {"name": "builtin"},
                "runtime": {"workers": 1, "ui": "none"},
                "workload": {
                    "type": "agentic",
                    "config": {
                        "dataset": "fixture/agentic@locked",
                        "evaluator": {
                            "python_executable": str(Path(sys.executable).resolve()),
                            "worker_module": "fixture_agentic_v2.worker",
                            "environment": {
                                "PYTHONPATH": str(worker_root),
                                "FIXTURE_ARTIFACT_TARGET": str(artifact_dir),
                            },
                        },
                        "task_concurrency": 1,
                        "environment": "fixture",
                        "output_dir": "episodes",
                        "max_tokens": 16,
                        "context_window": 4096,
                        "inference_gateway_host": "127.0.0.1",
                    },
                },
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
        benchmark_id="python-v2-agentic",
        cfg=config.benchmark,
        artifact_dir=artifact_dir,
        label="agentic",
        random_seed=43,
    )


def test_python_config_v2_reaches_direct_agentic_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agentic_installation: RunnerInstallation,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    worker_root = tmp_path / "worker"
    _install_worker(worker_root)
    with _AgenticChatHandler.lock:
        _AgenticChatHandler.bodies.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AgenticChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint_url = (
            f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        )
        run = _run(artifact_dir, endpoint_url, worker_root)
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
            artifact_dir,
            installation=agentic_installation,
        ).execute_sync(run)

        assert len(captured) == 1
        request, completed = captured[0]
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
        assert request["protocol_version"] == 2
        assert request["operation"] == "execute"
        assert request["run"]["transport"]["type"] == "http"
        workload = request["run"]["workload"]
        assert workload["type"] == "agentic"
        assert workload["config"]["dataset"] == "fixture/agentic@locked"
        assert isinstance(workload["config"]["dataset"], str)
        assert not hasattr(rust_executor, "build_run_request")
        assert not hasattr(RustSubprocessExecutor, "_resolve_run")

        assert result.success, result.error
        report = orjson.loads((artifact_dir / "native-v2.json").read_bytes())
        assert report["run"]["mode"] == "agentic_accuracy"
        assert report["run"]["transport"] == "http"
        assert report["run"]["workload"] == "agentic"
        assert report["agentic"]["summary"]["episode_count"] == 1
        assert report["agentic"]["summary"]["completed_count"] == 1
        assert report["agentic"]["summary"]["primary_score"] == 1.0
        with _AgenticChatHandler.lock:
            assert len(_AgenticChatHandler.bodies) == 1
            assert _AgenticChatHandler.bodies[0]["model"] == "primary-model"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
