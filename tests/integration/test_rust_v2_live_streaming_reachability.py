# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config v2 -> v2 runner -> post-artifact live-worker activation proof."""

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


def read_event():
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("unexpected EOF")
    return json.loads(line)


def reply(value):
    print(json.dumps(value), flush=True)


initialize = read_event()
if initialize.get("protocol_version") != 1 or initialize.get("event") != "initialize":
    raise RuntimeError(f"invalid initialize event: {initialize!r}")
if initialize["config"]["artifact_dir"] != ARTIFACT_DIR:
    raise RuntimeError("worker received the wrong artifact target")
if os.path.exists(ARTIFACT_DIR):
    raise RuntimeError("artifact target existed during worker preparation")
reply({
    "protocol_version": 1,
    "event": "prepared",
    "active": True,
    "disabled_reason": None,
})

activate = read_event()
if activate != {"protocol_version": 1, "event": "activate"}:
    raise RuntimeError(f"invalid activate event: {activate!r}")
if not os.path.isdir(ARTIFACT_DIR):
    raise RuntimeError("artifact target was absent during worker activation")
with open(os.path.join(ARTIFACT_DIR, "python-live-activation.json"), "w", encoding="utf-8") as output:
    json.dump({"activated_after_artifact_commit": True}, output)
reply({
    "protocol_version": 1,
    "event": "ready",
    "active": False,
    "disabled_reason": "fixture does not export",
})

shutdown = read_event()
if shutdown.get("protocol_version") != 1 or shutdown.get("event") != "shutdown":
    raise RuntimeError(f"invalid shutdown event: {shutdown!r}")
reply({
    "protocol_version": 1,
    "event": "terminal",
    "success": True,
    "metric_records": 0,
    "phase_events": 0,
    "processing_errors": 0,
    "dropped_events": shutdown["dropped_events"],
})
"""


class _ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests = 0
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        assert self.path == "/v1/chat/completions"
        with _ChatHandler.lock:
            _ChatHandler.requests += 1
        response = b"".join(
            [
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
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
def scheduled_installation() -> RunnerInstallation:
    installation = RunnerInstallation.resolve(_runner_binary())
    assert "http" in installation.capabilities.get("transport", {})
    return installation


def _install_worker(root: Path) -> None:
    package = root / "aiperf" / "post_processors"
    package.mkdir(parents=True)
    (root / "aiperf" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "native_streaming_worker.py").write_text(
        _WORKER_SOURCE,
        encoding="utf-8",
    )


def _run(artifact_dir: Path, endpoint_url: str) -> BenchmarkRun:
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
                    "entries": 2,
                    "isl": 2,
                    "osl": 1,
                },
                "profiling": {
                    "type": "concurrency",
                    "requests": 2,
                    "concurrency": 1,
                },
                "tokenizer": {"name": "builtin"},
                "runtime": {"workers": 1, "ui": "none"},
                "otel": {"metrics_url": "http://127.0.0.1:4318"},
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
        benchmark_id="python-v2-live-streaming",
        cfg=config.benchmark,
        artifact_dir=artifact_dir,
        label="live-streaming",
        random_seed=31,
    )


def test_python_config_v2_reaches_live_worker_without_v1_or_early_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheduled_installation: RunnerInstallation,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    worker_root = tmp_path / "worker"
    _install_worker(worker_root)
    monkeypatch.setenv("PYTHONPATH", str(worker_root))
    monkeypatch.setenv("FIXTURE_ARTIFACT_DIR", str(artifact_dir))

    with _ChatHandler.lock:
        _ChatHandler.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        run = _run(
            artifact_dir,
            f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions",
        )

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
            installation=scheduled_installation,
        ).execute_sync(run)

        assert len(captured) == 1
        request, completed = captured[0]
        assert request["protocol_version"] == 2
        assert request["run"]["cfg"]["datasets"][0]["type"] == "synthetic"
        live = request["run"]["cfg"]["sidecars"]["live_streaming"]
        assert live["worker_module"] == (
            "aiperf.post_processors.native_streaming_worker"
        )
        assert live["otel"]["metrics_url"] == ("http://127.0.0.1:4318/v1/metrics")

        terminal = orjson.loads(completed.stdout)
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
        assert terminal["success"] is True
        assert result.success, result.error
        assert result.summary_metrics["request_count"].avg == 2.0
        proof = orjson.loads(
            (artifact_dir / "python-live-activation.json").read_bytes()
        )
        assert proof == {"activated_after_artifact_commit": True}
        with _ChatHandler.lock:
            assert _ChatHandler.requests == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
