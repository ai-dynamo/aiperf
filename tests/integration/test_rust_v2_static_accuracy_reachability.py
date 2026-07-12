# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config v2 -> v2 runner -> canonical static-evaluator product proof."""

from __future__ import annotations

import copy
import os
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

_WORKER_SOURCE = r"""
import json
import os
import sys

ARTIFACT_DIR = os.environ["FIXTURE_ARTIFACT_DIR"]
load_count = 0
page_count = 0

PROBLEMS = [
    {
        "problem_id": "fixture-0",
        "task": "task-a",
        "prompt": "first fixture",
        "messages": [{"role": "user", "content": "first fixture"}],
        "generation": {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0, "stop": []},
    },
    {
        "problem_id": "fixture-1",
        "task": "task-b",
        "prompt": "second fixture",
        "messages": [{"role": "user", "content": "second fixture"}],
        "generation": {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0, "stop": []},
    },
]


def assert_unmaterialized(operation):
    if os.path.exists(ARTIFACT_DIR):
        raise RuntimeError(f"artifact root existed during {operation}")


for line in sys.stdin:
    request = json.loads(line)
    op = request["op"]
    if op == "hello":
        assert_unmaterialized(op)
        result = {
            "protocol": 1,
            "worker_version": "fixture-v2-1",
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": {"fixture-evaluator": "1"},
            "worker_source_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "container_digest": None,
            "capabilities": ["load", "next_problems", "grade_batch", "grader_override", "shutdown"],
        }
    elif op == "load":
        assert_unmaterialized(op)
        load_count += 1
        if load_count != 1:
            raise RuntimeError(f"load called {load_count} times")
        result = {
            "benchmark": request["benchmark"],
            "problem_count": len(PROBLEMS),
            "dataset": {
                "provider": "fixture",
                "benchmark": request["benchmark"],
                "repository": "fixture/repository",
                "subset": "default",
                "revision": "fixture-revision",
                "evaluation_splits": ["test"],
                "task_version": 1,
            },
            "grader": request.get("grader") or "fixture-python-grader",
        }
    elif op == "next_problems":
        assert_unmaterialized(op)
        page_count += 1
        start = request["offset"]
        end = min(start + request["limit"], len(PROBLEMS))
        result = {
            "items": PROBLEMS[start:end],
            "next_offset": end,
            "done": end == len(PROBLEMS),
        }
    elif op == "grade_batch":
        if not os.path.isdir(ARTIFACT_DIR):
            raise RuntimeError("artifact root absent during grading")
        if load_count != 1 or page_count != 1:
            raise RuntimeError(f"unexpected lifecycle load={load_count} pages={page_count}")
        result = {
            "items": [
                {
                    "problem_id": item["problem_id"],
                    "task": "task-a" if item["problem_id"] == "fixture-0" else "task-b",
                    "correct": item["problem_id"] == "fixture-0",
                    "unparsed": False,
                    "confidence": 1.0 if item["problem_id"] == "fixture-0" else 0.0,
                    "reasoning": "graded in fixture Python worker",
                    "extracted_answer": item["response"],
                }
                for item in request["items"]
            ]
        }
    elif op == "shutdown":
        if load_count != 1 or page_count != 1:
            raise RuntimeError(f"unexpected shutdown lifecycle load={load_count} pages={page_count}")
        result = {"shutdown": True}
    else:
        raise RuntimeError(op)
    print(json.dumps({"id": request["id"], "ok": True, "result": result}), flush=True)
    if op == "shutdown":
        break
"""


class _AccuracyChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    prompts: list[str] = []
    prompts_lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = orjson.loads(self.rfile.read(length))
        assert self.path == "/v1/chat/completions"
        prompt = body["messages"][-1]["content"]
        assert isinstance(prompt, str)
        with self.prompts_lock:
            self.prompts.append(prompt)
        response = b"".join(
            [
                b'data: {"choices":[{"delta":{"content":"fixture answer"}}]}\n\n',
                b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n',
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
def static_accuracy_installation() -> RunnerInstallation:
    installation = RunnerInstallation.resolve(_runner_binary())
    assert installation.supports_pair("online_http", "static_accuracy")
    return installation


def _accuracy_run(artifact_dir: Path, endpoint_url: str) -> BenchmarkRun:
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
                    "type": "synthetic",
                    "entries": 1,
                    "isl": 2,
                    "osl": 1,
                },
                "profiling": {
                    "type": "concurrency",
                    "requests": 2,
                    "concurrency": 2,
                },
                "accuracy": {
                    "benchmark": "gsm8k",
                    "grader": "exact_match",
                },
                "tokenizer": {"name": "builtin"},
                "runtime": {"workers": 1, "ui": "none"},
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
        benchmark_id="python-v2-static-accuracy",
        cfg=config.benchmark,
        artifact_dir=artifact_dir,
        label="static-accuracy",
        random_seed=29,
    )


def _install_fixture_worker(root: Path) -> None:
    package = root / "aiperf" / "accuracy"
    package.mkdir(parents=True)
    (root / "aiperf" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "worker.py").write_text(_WORKER_SOURCE, encoding="utf-8")


def test_python_config_v2_reaches_static_accuracy_without_v1_or_python_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    static_accuracy_installation: RunnerInstallation,
) -> None:
    worker_root = tmp_path / "worker"
    _install_fixture_worker(worker_root)
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setenv("PYTHONPATH", str(worker_root))
    monkeypatch.setenv("FIXTURE_ARTIFACT_DIR", str(artifact_dir))

    with _AccuracyChatHandler.prompts_lock:
        _AccuracyChatHandler.prompts.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AccuracyChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint_url = (
            f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        )
        run = _accuracy_run(artifact_dir, endpoint_url)

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
            installation=static_accuracy_installation,
        ).execute_sync(run)

        assert len(captured) == 1
        request, completed = captured[0]
        assert request["protocol_version"] == 2
        assert request["operation"] == "execute"
        assert request["run"]["backend"] == {
            "type": "online_http",
            "config": {},
        }
        assert request["run"]["workload"]["type"] == "static_accuracy"
        accuracy = request["run"]["workload"]["config"]["accuracy"]
        assert accuracy["benchmark"] == "gsm8k"
        assert accuracy["worker_module"] == "aiperf.accuracy.worker"
        assert Path(accuracy["python_executable"]).is_absolute()

        terminal = orjson.loads(completed.stdout)
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
        assert terminal["success"] is True
        assert terminal["provenance"] == {
            "backend": "online_http",
            "workload": "static_accuracy",
        }
        assert result.success, result.error
        assert result.summary_metrics["request_count"].avg == 2.0

        native = orjson.loads((artifact_dir / "native-v2.json").read_bytes())
        assert native["run"]["mode"] == "accuracy"
        assert native["run"]["backend"] == "online_http"
        assert native["run"]["workload"] == "static_accuracy"
        overall = native["accuracy"]["summary"]["overall"]
        assert overall["accuracy"] == 0.5
        assert overall["correct_count"] == 1
        assert overall["n"] == 2
        assert native["evaluator"]["dataset"]["revision"] == "fixture-revision"

        with _AccuracyChatHandler.prompts_lock:
            prompts = sorted(_AccuracyChatHandler.prompts)
        assert prompts == ["first fixture", "second fixture"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
