# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python Config v2 -> stdio -> Rust HTTP/SSE -> native-v2 proof."""

from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import orjson

from aiperf.common.models.record_models import MetricRecordInfo
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.orchestrator.jsonl_loader import load_single_metric
from aiperf.orchestrator.rust_executor import RustSubprocessExecutor

_SSE = b"".join(
    [
        b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}]}\n\n',
        b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":1}}\n\n',
        b"data: [DONE]\n\n",
    ]
)


class _ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(_SSE)))
        self.end_headers()
        self.wfile.write(_SSE)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_config_v2_executes_a_real_native_child(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        envelope = AIPerfConfig.model_validate(
            {
                "benchmark": {
                    "models": ["mock-model"],
                    "endpoint": {
                        "urls": [f"http://127.0.0.1:{port}/v1/chat/completions"],
                        "streaming": True,
                        "use_server_token_count": True,
                    },
                    "dataset": {
                        "type": "synthetic",
                        "entries": 4,
                        "isl": 8,
                        "osl": 1,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "requests": 4,
                        "concurrency": 2,
                    },
                    "artifacts": {"dir": str(tmp_path)},
                    "slos": {"request_latency": 1000},
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        run = BenchmarkRun(
            benchmark_id="python-rust-e2e",
            cfg=envelope.benchmark,
            artifact_dir=tmp_path,
            label="native",
            random_seed=11,
        )
        default_binary = Path(__file__).parents[2] / "target/debug/aiperf-runner"
        binary = Path(os.environ.get("AIPERF_RUNNER_BIN", default_binary))
        result = RustSubprocessExecutor(tmp_path, binary=binary).execute_sync(run)

        assert result.success, result.error
        assert result.summary_metrics["request_count"].avg == 4.0
        assert result.summary_metrics["total_output_tokens"].avg == 4.0
        assert result.summary_metrics["request_latency"].count == 4
        assert result.summary_metrics["good_request_count"].avg == 4.0
        assert (tmp_path / "native-v2.json").is_file()
        records_path = tmp_path / "profile_export.jsonl"
        rows = [orjson.loads(line) for line in records_path.read_bytes().splitlines()]
        assert len(rows) == 4
        assert all(MetricRecordInfo.model_validate(row) for row in rows)
        assert all(row["metadata"]["benchmark_phase"] == "profiling" for row in rows)
        assert all("time_to_first_token" in row["metrics"] for row in rows)
        assert len(load_single_metric(tmp_path, "request_latency")) == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
