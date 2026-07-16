# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python Config v2 -> stdio -> Rust HTTP/SSE -> native-v2 proof."""

from __future__ import annotations

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import orjson
import pytest

from aiperf.common.models import NetworkLatencySample
from aiperf.common.models.record_models import MetricRecordInfo, RawRecordInfo
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.orchestrator import native_execution
from aiperf.orchestrator.jsonl_loader import load_single_metric
from aiperf.orchestrator.native_execution import NativeExecutor

_SSE = b"".join(
    [
        b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}]}\n\n',
        b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":1}}\n\n',
        b"data: [DONE]\n\n",
    ]
)

_NON_STREAMING = orjson.dumps(
    {
        "id": "connection-proof",
        "object": "chat.completion",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "a"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 1},
    }
)

_SHAREGPT = orjson.dumps(
    [
        {
            "conversations": [
                {"from": "human", "value": "one two three four five"},
                {"from": "gpt", "value": "alpha beta gamma delta epsilon"},
            ]
        }
    ]
)

_WORDLEVEL_TOKENIZER = """{
  "version":"1.0",
  "truncation":null,
  "padding":null,
  "added_tokens":[
    {"id":0,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
    {"id":1,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
    {"id":2,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
  ],
  "normalizer":null,
  "pre_tokenizer":{"type":"Whitespace"},
  "post_processor":null,
  "decoder":null,
  "model":{"type":"WordLevel","vocab":{"[UNK]":0,"<s>":1,"</s>":2,"user":3,"hello":4,"assistant":5},"unk_token":"[UNK]"}
}"""

_WORDLEVEL_CONFIG = """{
  "bos_token":"<s>",
  "eos_token":"</s>",
  "chat_template":"{{ bos_token }} {% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
}"""


def _assert_protocol_v2_only() -> None:
    assert not hasattr(native_execution, "validate_v1_selection")
    assert not hasattr(native_execution, "build_run_request")
    assert not hasattr(NativeExecutor, "_resolve_run")


class _ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    bodies: list[dict[str, object]] = []
    telemetry_scrapes = 0
    telemetry_lock = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/metrics":
            with self.telemetry_lock:
                type(self).telemetry_scrapes += 1
                scrape = type(self).telemetry_scrapes
            body = (
                'DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-python-e2e",modelName="H100",Hostname="node"} 250\n'
                'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0",UUID="GPU-python-e2e",modelName="H100",Hostname="node"} '
                f"{scrape * 1_000_000_000}\n"
            ).encode()
        elif self.path == "/dataset/sharegpt.json":
            body = _SHAREGPT
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/plain" if self.path == "/metrics" else "application/json",
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.bodies.append(orjson.loads(body))
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


class _AdaptiveChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    active = 0
    peak = 0
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        with self.lock:
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
        try:
            time.sleep(0.05)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(_SSE)))
            self.end_headers()
            self.wfile.write(_SSE)
        finally:
            with self.lock:
                type(self).active -= 1

    def log_message(self, format: str, *args: object) -> None:
        pass


class _ConnectionHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    peer_ports: list[int] = []
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        with self.lock:
            self.peer_ports.append(self.client_address[1])
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(_NON_STREAMING)))
        self.end_headers()
        self.wfile.write(_NON_STREAMING)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _OTLPSinkHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    exports: list[bytes] = []
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path != "/v1/metrics":
            self.send_error(404)
            return
        with self.lock:
            self.exports.append(body)
        response = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _ServerMetricsHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    primary_scrapes = 0
    prometheus_scrapes = 0
    lock = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/chat/completions/metrics":
            with self.lock:
                type(self).primary_scrapes += 1
            body = b'{"iteration_stats": []}'
            content_type = "application/json"
        elif self.path == "/v1/chat/completions/prometheus/metrics":
            with self.lock:
                type(self).prometheus_scrapes += 1
                scrape = type(self).prometheus_scrapes
            body = "\n".join(
                [
                    "# HELP requests_total Completed requests",
                    "# TYPE requests_total counter",
                    f'requests_total{{status="ok"}} {scrape * 10}',
                    "# HELP kv_cache_usage_ratio KV cache usage",
                    "# TYPE kv_cache_usage_ratio gauge",
                    f'kv_cache_usage_ratio{{model="mock-model"}} {scrape / 10}',
                    "# HELP request_latency_seconds Request latency",
                    "# TYPE request_latency_seconds histogram",
                    f'request_latency_seconds_bucket{{le="0.1"}} {scrape}',
                    f'request_latency_seconds_bucket{{le="0.5"}} {scrape * 2}',
                    f'request_latency_seconds_bucket{{le="+Inf"}} {scrape * 3}',
                    f"request_latency_seconds_sum {scrape * 0.9}",
                    f"request_latency_seconds_count {scrape * 3}",
                    "",
                ]
            ).encode()
            content_type = "text/plain; version=0.0.4"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        time.sleep(0.03)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(_SSE)))
        self.end_headers()
        self.wfile.write(_SSE)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_config_v2_streams_rust_metrics_live_through_canonical_python_otel(
    tmp_path: Path,
) -> None:
    # This test pins the legacy live-streaming Python OTel sidecar (mid-run OTLP
    # emission). On the default native path that sidecar is suppressed and the
    # native aiperf::export::otel sink emits post-run instead, so mid-run arrival
    # is not observable. Exercise the still-supported legacy path via
    # AIPERF_RUNTIME_NATIVE_EXPORT=0 (mirrors AIPERF_RUNTIME_ENGINE=python A/B).
    from aiperf.common.environment import Environment

    metrics_service_pb2 = pytest.importorskip(
        "opentelemetry.proto.collector.metrics.v1.metrics_service_pb2"
    )
    export_request_type = metrics_service_pb2.ExportMetricsServiceRequest

    _AdaptiveChatHandler.active = 0
    _AdaptiveChatHandler.peak = 0
    _OTLPSinkHandler.exports = []
    inference_server = ThreadingHTTPServer(("127.0.0.1", 0), _AdaptiveChatHandler)
    otlp_server = ThreadingHTTPServer(("127.0.0.1", 0), _OTLPSinkHandler)
    inference_thread = threading.Thread(
        target=inference_server.serve_forever, daemon=True
    )
    otlp_thread = threading.Thread(target=otlp_server.serve_forever, daemon=True)
    inference_thread.start()
    otlp_thread.start()
    try:
        inference_port = inference_server.server_address[1]
        otlp_port = otlp_server.server_address[1]
        envelope = AIPerfConfig.model_validate(
            {
                "benchmark": {
                    "models": ["mock-model"],
                    "endpoint": {
                        "urls": [
                            f"http://127.0.0.1:{inference_port}/v1/chat/completions"
                        ],
                        "streaming": True,
                        "use_server_token_count": True,
                    },
                    "dataset": {
                        "type": "synthetic",
                        "entries": 80,
                        "isl": 8,
                        "osl": 1,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "requests": 80,
                        "concurrency": 2,
                    },
                    "artifacts": {"dir": str(tmp_path), "records": False},
                    "otel": {
                        "metrics_url": f"http://127.0.0.1:{otlp_port}",
                        "custom_resource_attributes": {
                            "team": "native-e2e",
                            "path": "rust-stdio-python",
                        },
                    },
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        run = BenchmarkRun(
            benchmark_id="python-rust-live-otel-e2e",
            cfg=envelope.benchmark,
            artifact_dir=tmp_path,
            label="native-live-otel",
            random_seed=43,
        )
        default_binary = Path(__file__).parents[2] / "target/debug/aiperf runner"
        binary = Path(os.environ.get("AIPERF_RUNNER_BIN", default_binary))
        result_holder: dict[str, object] = {}

        def execute() -> None:
            result_holder["result"] = NativeExecutor(
                tmp_path, binary=binary
            ).execute_sync(run)

        with (
            mock.patch.dict(
                os.environ,
                {"AIPERF_OTEL_FLUSH_INTERVAL_SECONDS": "0.1"},
            ),
            mock.patch.object(Environment.RUNTIME, "NATIVE_EXPORT", False),
        ):
            runner_thread = threading.Thread(target=execute)
            runner_thread.start()
            deadline = time.monotonic() + 60.0
            export_arrived_mid_run = False
            while time.monotonic() < deadline:
                with _OTLPSinkHandler.lock:
                    has_export = bool(_OTLPSinkHandler.exports)
                if has_export:
                    export_arrived_mid_run = runner_thread.is_alive()
                    break
                if not runner_thread.is_alive():
                    break
                time.sleep(0.01)
            runner_thread.join(timeout=120)

        assert not runner_thread.is_alive(), "native runner did not terminate"
        result = result_holder["result"]
        assert result.success, result.error
        assert export_arrived_mid_run, (
            "OTLP received no Rust-owned metrics before the native child completed"
        )
        assert result.summary_metrics["request_count"].avg == 80.0
        assert not (tmp_path / "profile_export.jsonl").exists()

        with _OTLPSinkHandler.lock:
            payloads = list(_OTLPSinkHandler.exports)
        exports = []
        for payload in payloads:
            request = export_request_type()
            request.ParseFromString(payload)
            exports.append(request)
        assert exports

        resource_attributes: dict[str, str] = {}
        metric_names: set[str] = set()
        histogram_points = 0
        for request in exports:
            for resource_metrics in request.resource_metrics:
                for attribute in resource_metrics.resource.attributes:
                    field = attribute.value.WhichOneof("value")
                    if field is not None:
                        resource_attributes[attribute.key] = str(
                            getattr(attribute.value, field)
                        )
                for scope_metrics in resource_metrics.scope_metrics:
                    for metric in scope_metrics.metrics:
                        metric_names.add(metric.name)
                        if metric.HasField("histogram"):
                            histogram_points += len(metric.histogram.data_points)

        assert resource_attributes["service.name"] == "aiperf"
        assert resource_attributes["aiperf.benchmark.id"] == "python-rust-live-otel-e2e"
        assert resource_attributes["team"] == "native-e2e"
        assert resource_attributes["path"] == "rust-stdio-python"
        assert "aiperf.timing.requests.sent" in metric_names
        assert any(
            name.startswith("aiperf.") or name.startswith("gen_ai.")
            for name in metric_names
        )
        assert histogram_points > 0
    finally:
        inference_server.shutdown()
        inference_server.server_close()
        inference_thread.join(timeout=5)
        otlp_server.shutdown()
        otlp_server.server_close()
        otlp_thread.join(timeout=5)


def test_config_v2_collects_server_metrics_in_rust_across_exact_phase_boundaries(
    tmp_path: Path,
) -> None:
    _assert_protocol_v2_only()
    import pyarrow.parquet as pq

    from aiperf.common.environment import Environment
    from aiperf.common.models.server_metrics_models import SlimRecord

    _ServerMetricsHandler.primary_scrapes = 0
    _ServerMetricsHandler.prometheus_scrapes = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ServerMetricsHandler)
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
                        "entries": 5,
                        "isl": 8,
                        "osl": 1,
                    },
                    "phases": [
                        {
                            "name": "warmup",
                            "type": "concurrency",
                            "requests": 2,
                            "concurrency": 1,
                        },
                        {
                            "name": "profiling",
                            "type": "concurrency",
                            "requests": 3,
                            "concurrency": 1,
                        },
                    ],
                    "artifacts": {
                        "dir": str(tmp_path),
                        "slice_duration": 0.02,
                    },
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"formats": ["json", "csv", "jsonl", "parquet"]},
                    "runtime": {"ui": "none"},
                }
            }
        )
        run = BenchmarkRun(
            benchmark_id="python-rust-server-metrics-e2e",
            cfg=envelope.benchmark,
            artifact_dir=tmp_path,
            label="native-server-metrics",
            random_seed=31,
        )
        default_binary = Path(__file__).parents[2] / "target/debug/aiperf runner"
        binary = Path(os.environ.get("AIPERF_RUNNER_BIN", default_binary))

        with mock.patch.object(Environment.SERVER_METRICS, "COLLECTION_INTERVAL", 0.01):
            result = NativeExecutor(tmp_path, binary=binary).execute_sync(run)

        assert result.success, result.error
        assert result.summary_metrics["request_count"].avg == 3.0
        assert _ServerMetricsHandler.primary_scrapes == 1
        assert _ServerMetricsHandler.prometheus_scrapes >= 6

        native = orjson.loads((tmp_path / "native-v2.json").read_bytes())
        metadata = native["summary"]["server_metrics"]
        assert metadata["profiling"]["start_ns"] < metadata["profiling"]["end_ns"]
        assert metadata["warmup"]["start_ns"] < metadata["warmup"]["end_ns"]
        assert metadata["endpoints_successful"] == [
            f"http://127.0.0.1:{port}/v1/chat/completions/prometheus/metrics"
        ]
        assert set(native["server_metrics"]) == {
            "kv_cache_usage_ratio",
            "request_latency_seconds",
            "requests",
        }
        assert set(native["warmup_server_metrics"]) == set(native["server_metrics"])
        assert native["server_metrics"]["requests"]["type"] == "counter"
        assert native["server_metrics"]["requests"]["series"][0]["stats"]["total"] > 0
        assert (
            native["server_metrics"]["request_latency_seconds"]["type"] == "histogram"
        )
        assert (
            native["server_metrics"]["request_latency_seconds"]["series"][0]["stats"][
                "count"
            ]
            > 0
        )

        slim_rows = [
            SlimRecord.model_validate(orjson.loads(line))
            for line in (tmp_path / "server_metrics_export.jsonl")
            .read_bytes()
            .splitlines()
        ]
        assert len(slim_rows) >= 4
        assert {str(row.benchmark_phase) for row in slim_rows} == {
            "warmup",
            "profiling",
        }
        assert all("requests" in row.metrics for row in slim_rows)

        compatibility = orjson.loads(
            (tmp_path / "server_metrics_export.json").read_bytes()
        )
        assert compatibility["metrics"]["requests"]["type"] == "counter"
        assert compatibility["warmup_metrics"]["requests"]["type"] == "counter"
        assert compatibility["metrics"]["request_latency_seconds"]["type"] == (
            "histogram"
        )
        assert "requests" in (tmp_path / "server_metrics_export.csv").read_text()

        parquet_path = tmp_path / "server_metrics_export.parquet"
        table = pq.read_table(parquet_path)
        assert table.num_rows > 0
        assert set(table.column("metric_type").to_pylist()) == {
            "counter",
            "gauge",
            "histogram",
        }
        assert not (tmp_path / ".aiperf-server-metrics-parquet-wire.jsonl").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_config_v2_joins_rust_gpu_telemetry_into_all_artifacts(
    tmp_path: Path,
) -> None:
    # The telemetry_data block embedded in profile_export_aiperf.json is a
    # Python-ExporterManager artifact; the native genai_perf sink omits it by
    # design (GPU telemetry lives in native-v2.json series, the CSV GPU rows, and
    # gpu_telemetry_export.jsonl). This test pins the legacy Python compat
    # renderer via AIPERF_RUNTIME_NATIVE_EXPORT=0 to assert the telemetry_data
    # projection end-to-end (mirrors AIPERF_RUNTIME_ENGINE=python A/B).
    from aiperf.common.environment import Environment

    _assert_protocol_v2_only()
    _ChatHandler.bodies.clear()
    _ChatHandler.telemetry_scrapes = 0
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
                    "gpu_telemetry": {"urls": [f"http://127.0.0.1:{port}/metrics"]},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        run = BenchmarkRun(
            benchmark_id="python-rust-gpu-e2e",
            cfg=envelope.benchmark,
            artifact_dir=tmp_path,
            label="native-gpu",
            random_seed=19,
        )
        default_binary = Path(__file__).parents[2] / "target/debug/aiperf runner"
        binary = Path(os.environ.get("AIPERF_RUNNER_BIN", default_binary))

        with mock.patch.object(Environment.RUNTIME, "NATIVE_EXPORT", False):
            result = NativeExecutor(tmp_path, binary=binary).execute_sync(run)

        assert result.success, result.error
        assert result.summary_metrics["request_count"].avg == 4.0
        assert result.summary_metrics["total_gpu_power"].avg == 250.0
        assert result.summary_metrics["total_gpu_energy"].avg > 0.0
        assert _ChatHandler.telemetry_scrapes >= 2

        native = orjson.loads((tmp_path / "native-v2.json").read_bytes())
        gpu_series = native["metrics"]["gpu_power_usage"]["series"][0]
        assert gpu_series["labels"]["gpu_uuid"] == "GPU-python-e2e"
        assert gpu_series["stats"]["avg"] == 250.0

        compatibility = orjson.loads(
            (tmp_path / "profile_export_aiperf.json").read_bytes()
        )
        endpoint = compatibility["telemetry_data"]["endpoints"][f"127.0.0.1:{port}"]
        gpu = endpoint["gpus"]["gpu_0"]
        assert gpu["gpu_uuid"] == "GPU-python-e2e"
        assert gpu["metrics"]["gpu_power_usage"]["avg"] == 250.0

        rows = [
            orjson.loads(line)
            for line in (tmp_path / "gpu_telemetry_export.jsonl")
            .read_bytes()
            .splitlines()
        ]
        assert len(rows) >= 2
        assert all(row["telemetry_data"]["gpu_power_usage"] == 250.0 for row in rows)
        csv = (tmp_path / "profile_export_aiperf.csv").read_text()
        assert "GPU Power Usage (W)" in csv
        assert "GPU-python-e2e" in csv
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_config_v2_runs_native_tcp_rtt_calibration_and_adjusts_metrics(
    tmp_path: Path,
) -> None:
    _assert_protocol_v2_only()
    _ChatHandler.bodies.clear()
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
                        "entries": 2,
                        "isl": 8,
                        "osl": 1,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "requests": 2,
                        "concurrency": 1,
                    },
                    "network_latency": {
                        "enabled": True,
                        "ping_interval": 0.01,
                    },
                    "artifacts": {"dir": str(tmp_path)},
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        run = BenchmarkRun(
            benchmark_id="python-rust-network-rtt-e2e",
            cfg=envelope.benchmark,
            artifact_dir=tmp_path,
            label="native-network-rtt",
            random_seed=23,
        )
        default_binary = Path(__file__).parents[2] / "target/debug/aiperf runner"
        binary = Path(os.environ.get("AIPERF_RUNNER_BIN", default_binary))

        result = NativeExecutor(tmp_path, binary=binary).execute_sync(run)

        assert result.success, result.error
        assert result.summary_metrics["request_count"].avg == 2.0
        assert result.summary_metrics["network_rtt"].avg > 0.0
        assert "network_adjusted_request_latency" in result.summary_metrics
        assert "network_adjusted_time_to_first_token" in result.summary_metrics

        rows = [
            NetworkLatencySample.model_validate(orjson.loads(line))
            for line in (tmp_path / "profile_export_network_latency.jsonl")
            .read_bytes()
            .splitlines()
        ]
        assert len(rows) >= 5
        assert sum(sample.success for sample in rows) >= 5
        assert all(sample.probe_type == "tcp_connect" for sample in rows)
        assert all(sample.target_host == "127.0.0.1" for sample in rows)
        assert all(sample.target_port == port for sample in rows)

        compatibility = orjson.loads(
            (tmp_path / "profile_export_aiperf.json").read_bytes()
        )
        assert compatibility["network_rtt"]["avg"] > 0.0
        assert "network_adjusted_request_latency" in compatibility
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_config_v2_fixed_network_rtt_bypasses_probes_and_shifts_metrics(
    tmp_path: Path,
) -> None:
    _assert_protocol_v2_only()
    _AdaptiveChatHandler.active = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AdaptiveChatHandler)
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
                        "entries": 2,
                        "isl": 8,
                        "osl": 1,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "requests": 2,
                        "concurrency": 1,
                    },
                    "network_latency": {"enabled": True, "mean_ms": 5.0},
                    "artifacts": {"dir": str(tmp_path)},
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        run = BenchmarkRun(
            benchmark_id="python-rust-fixed-network-rtt-e2e",
            cfg=envelope.benchmark,
            artifact_dir=tmp_path,
            label="native-fixed-network-rtt",
            random_seed=29,
        )
        default_binary = Path(__file__).parents[2] / "target/debug/aiperf runner"
        binary = Path(os.environ.get("AIPERF_RUNNER_BIN", default_binary))

        result = NativeExecutor(tmp_path, binary=binary).execute_sync(run)

        assert result.success, result.error
        assert result.summary_metrics["network_rtt"].avg == 5.0
        raw = result.summary_metrics["request_latency"]
        adjusted = result.summary_metrics["network_adjusted_request_latency"]
        assert raw.avg - adjusted.avg == 5.0
        assert raw.std == adjusted.std
        assert not (tmp_path / "profile_export_network_latency.jsonl").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_config_v2_executes_a_real_native_child(tmp_path: Path) -> None:
    _ChatHandler.bodies.clear()
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
                        "api_key": "config-v2-raw-secret",
                        "headers": {"X-Custom-Tracking": "config-v2-trace"},
                        "session_header": "X-Session-ID",
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
                    "artifacts": {"dir": str(tmp_path), "raw": True},
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
        default_binary = Path(__file__).parents[2] / "target/debug/aiperf runner"
        binary = Path(os.environ.get("AIPERF_RUNNER_BIN", default_binary))
        result = NativeExecutor(tmp_path, binary=binary).execute_sync(run)

        assert result.success, result.error
        assert result.summary_metrics["request_count"].avg == 4.0
        assert result.summary_metrics["total_output_tokens"].avg == 4.0
        assert result.summary_metrics["request_latency"].count == 4
        assert result.summary_metrics["good_request_count"].avg == 4.0
        assert (tmp_path / "native-v2.json").is_file()
        compatibility = orjson.loads(
            (tmp_path / "profile_export_aiperf.json").read_bytes()
        )
        assert compatibility["request_count"]["avg"] == 4.0
        assert compatibility["request_latency"]["count"] == 4
        assert compatibility["run_info"]["benchmark_id"] == "python-rust-e2e"
        assert (tmp_path / "profile_export_aiperf.csv").is_file()
        records_path = tmp_path / "profile_export.jsonl"
        rows = [orjson.loads(line) for line in records_path.read_bytes().splitlines()]
        assert len(rows) == 4
        assert all(MetricRecordInfo.model_validate(row) for row in rows)
        assert all(row["metadata"]["benchmark_phase"] == "profiling" for row in rows)
        assert all("time_to_first_token" in row["metrics"] for row in rows)
        assert len(load_single_metric(tmp_path, "request_latency")) == 4
        raw_bytes = (tmp_path / "profile_export_raw.jsonl").read_bytes()
        assert b"config-v2-raw-secret" not in raw_bytes
        raw_rows = [orjson.loads(line) for line in raw_bytes.splitlines()]
        assert len(raw_rows) == 4
        assert all(RawRecordInfo.model_validate(row) for row in raw_rows)
        assert all(
            row["request_headers"]["Authorization"] == "<redacted>"
            and row["request_headers"]["X-Custom-Tracking"] == "config-v2-trace"
            and row["request_headers"]["X-Session-ID"]
            == row["metadata"]["x_correlation_id"]
            and "X-Correlation-ID" not in row["request_headers"]
            and row["request_headers"]["X-Request-ID"]
            == row["metadata"]["x_request_id"]
            and row["status"] == 200
            and row["response_headers"]["content-type"] == "text/event-stream"
            and len(row["responses"]) == 3
            for row in raw_rows
        )
        assert sorted(orjson.dumps(row["payload"]) for row in raw_rows) == sorted(
            orjson.dumps(body) for body in _ChatHandler.bodies
        )

        cli_artifacts = tmp_path / "single-cli-run"
        cli_envelope = AIPerfConfig.model_validate(
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
                        "entries": 1,
                        "isl": 4,
                        "osl": 1,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "requests": 1,
                        "concurrency": 1,
                    },
                    "artifacts": {"dir": str(cli_artifacts)},
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        cli_run = BenchmarkRun(
            benchmark_id="python-cli-rust-e2e",
            cfg=cli_envelope.benchmark,
            artifact_dir=cli_artifacts,
            label="native-cli",
            random_seed=111,
        )
        from aiperf.cli_runner import _run_single_benchmark

        with (
            mock.patch.dict(os.environ, {"AIPERF_RUNNER_BIN": str(binary)}),
            mock.patch("os._exit") as exit_process,
        ):
            _run_single_benchmark(cli_run)

        exit_process.assert_called_once_with(0)
        assert (cli_artifacts / "native-v2.json").is_file()

        dataset_path = tmp_path / "file-dataset.jsonl"
        dataset_path.write_bytes(
            b'{"text":"first","output_length":1}\n{"text":"second","output_length":1}\n'
        )
        file_artifacts = tmp_path / "file-run"
        file_envelope = AIPerfConfig.model_validate(
            {
                "benchmark": {
                    "models": ["mock-model"],
                    "endpoint": {
                        "urls": [f"http://127.0.0.1:{port}/v1/chat/completions"],
                        "streaming": True,
                        "use_server_token_count": True,
                    },
                    "dataset": {
                        "type": "file",
                        "path": str(dataset_path),
                        "format": "single_turn",
                        "sampling": "sequential",
                    },
                    "profiling": {
                        "type": "concurrency",
                        "requests": 2,
                        "concurrency": 1,
                    },
                    "artifacts": {"dir": str(file_artifacts)},
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        file_run = BenchmarkRun(
            benchmark_id="python-rust-file-e2e",
            cfg=file_envelope.benchmark,
            artifact_dir=file_artifacts,
            label="native-file",
            random_seed=12,
        )
        file_result = NativeExecutor(
            file_artifacts, binary=binary
        ).execute_sync(file_run)

        assert file_result.success, file_result.error
        assert file_result.summary_metrics["request_count"].avg == 2.0
        assert file_result.summary_metrics["input_sequence_length"].count == 2
        assert (
            len((file_artifacts / "profile_export.jsonl").read_text().splitlines()) == 2
        )

        tokenizer_dir = tmp_path / "wordlevel-tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.json").write_text(_WORDLEVEL_TOKENIZER)
        (tokenizer_dir / "tokenizer_config.json").write_text(_WORDLEVEL_CONFIG)
        template_artifacts = tmp_path / "template-run"
        template_envelope = AIPerfConfig.model_validate(
            {
                "benchmark": {
                    "models": ["mock-model"],
                    "endpoint": {
                        "urls": [f"http://127.0.0.1:{port}/v1/chat/completions"],
                        "streaming": True,
                        "use_server_token_count": False,
                    },
                    "dataset": {
                        "type": "file",
                        "format": "single_turn",
                        "records": [{"text": "hello", "output_length": 1}],
                    },
                    "tokenizer": {
                        "name": str(tokenizer_dir),
                        "apply_chat_template": True,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "requests": 1,
                        "concurrency": 1,
                    },
                    "artifacts": {"dir": str(template_artifacts)},
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        template_run = BenchmarkRun(
            benchmark_id="python-rust-template-e2e",
            cfg=template_envelope.benchmark,
            artifact_dir=template_artifacts,
            label="native-template",
            random_seed=14,
        )
        template_result = NativeExecutor(
            template_artifacts, binary=binary
        ).execute_sync(template_run)

        assert template_result.success, template_result.error
        assert template_result.summary_metrics["input_sequence_length"].avg == 4.0
        template_row = orjson.loads(
            (template_artifacts / "profile_export.jsonl").read_bytes().splitlines()[0]
        )
        assert template_row["metrics"]["input_sequence_length"]["value"] == 4.0

        multimodal_artifacts = tmp_path / "multimodal-run"
        multimodal_envelope = AIPerfConfig.model_validate(
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
                        "entries": 1,
                        "random_seed": 19,
                        "sampling": "shuffle",
                        "prompts": {
                            "sequence_distribution": [
                                {"isl": 6, "osl": 3, "probability": 100}
                            ]
                        },
                        "prefix_prompts": {
                            "shared_system_length": 2,
                            "user_context_length": 2,
                        },
                        "images": {
                            "batch_size": 1,
                            "width": 4,
                            "height": 3,
                            "format": "png",
                            "source": "noise",
                        },
                        "audio": {
                            "batch_size": 1,
                            "length": 0.02,
                            "format": "wav",
                            "sample_rates": [8.0],
                            "depths": [8],
                            "channels": 1,
                        },
                    },
                    "profiling": {
                        "type": "concurrency",
                        "requests": 1,
                        "concurrency": 1,
                    },
                    "artifacts": {"dir": str(multimodal_artifacts)},
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        multimodal_run = BenchmarkRun(
            benchmark_id="python-rust-multimodal-e2e",
            cfg=multimodal_envelope.benchmark,
            artifact_dir=multimodal_artifacts,
            label="native-multimodal",
            random_seed=13,
        )
        multimodal_result = NativeExecutor(
            multimodal_artifacts, binary=binary
        ).execute_sync(multimodal_run)

        assert multimodal_result.success, multimodal_result.error
        assert multimodal_result.summary_metrics["request_count"].avg == 1.0
        encoded_body = orjson.dumps(_ChatHandler.bodies[-1])
        assert b'"max_completion_tokens":3' in encoded_body
        assert b'"image_url"' in encoded_body
        assert b'"input_audio"' in encoded_body

        from aiperf.dataset.loader.sharegpt import ShareGPTLoader

        public_artifacts = tmp_path / "public-run"
        public_envelope = AIPerfConfig.model_validate(
            {
                "benchmark": {
                    "models": ["mock-model"],
                    "endpoint": {
                        "urls": [f"http://127.0.0.1:{port}/v1/chat/completions"],
                        "streaming": True,
                        "use_server_token_count": True,
                    },
                    "dataset": {
                        "type": "public",
                        "dataset": "sharegpt",
                        "entries": 1,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "requests": 1,
                        "concurrency": 1,
                    },
                    "artifacts": {"dir": str(public_artifacts)},
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        public_run = BenchmarkRun(
            benchmark_id="python-rust-public-e2e",
            cfg=public_envelope.benchmark,
            artifact_dir=public_artifacts,
            label="native-public",
            random_seed=29,
        )
        local_dataset_url = f"http://127.0.0.1:{port}/dataset/sharegpt.json"
        with mock.patch.object(ShareGPTLoader, "url", local_dataset_url):
            public_result = NativeExecutor(
                public_artifacts, binary=binary
            ).execute_sync(public_run)

        assert public_result.success, public_result.error
        assert public_result.summary_metrics["request_count"].avg == 1.0
        assert b"one two three four five" in orjson.dumps(_ChatHandler.bodies[-1])

        synthesis_artifacts = tmp_path / "synthesis-run"
        synthesis_envelope = AIPerfConfig.model_validate(
            {
                "benchmark": {
                    "models": ["mock-model"],
                    "endpoint": {
                        "urls": [f"http://127.0.0.1:{port}/v1/chat/completions"],
                        "streaming": True,
                        "use_server_token_count": True,
                    },
                    "dataset": {
                        "type": "file",
                        "format": "mooncake_trace",
                        "records": [
                            {
                                "session_id": "trace-a",
                                "timestamp": 100,
                                "input_length": 1025,
                                "output_length": 2,
                                "hash_ids": [1, 2],
                            },
                            {
                                "session_id": "trace-b",
                                "timestamp": 202,
                                "input_length": 1025,
                                "output_length": 3,
                                "hash_ids": [1, 3],
                            },
                        ],
                        "synthesis": {
                            "speedup_ratio": 2,
                            "prefix_len_multiplier": 2,
                            "prompt_len_multiplier": 1,
                            "output_len_multiplier": 1.5,
                        },
                    },
                    "profiling": {
                        "type": "fixed_schedule",
                        "requests": 2,
                    },
                    "artifacts": {"dir": str(synthesis_artifacts)},
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        synthesis_run = BenchmarkRun(
            benchmark_id="python-rust-synthesis-e2e",
            cfg=synthesis_envelope.benchmark,
            artifact_dir=synthesis_artifacts,
            label="native-synthesis",
            random_seed=31,
        )
        body_start = len(_ChatHandler.bodies)
        synthesis_result = NativeExecutor(
            synthesis_artifacts, binary=binary
        ).execute_sync(synthesis_run)

        assert synthesis_result.success, synthesis_result.error
        assert synthesis_result.summary_metrics["request_count"].avg == 2.0
        synthesis_bodies = _ChatHandler.bodies[body_start:]
        assert [body["max_completion_tokens"] for body in synthesis_bodies] == [3, 4]
        synthesis_rows = [
            orjson.loads(line)
            for line in (synthesis_artifacts / "profile_export.jsonl")
            .read_bytes()
            .splitlines()
        ]
        start_delta_ns = (
            synthesis_rows[1]["metadata"]["request_start_ns"]
            - synthesis_rows[0]["metadata"]["request_start_ns"]
        )
        # The schedule is anchored before the first connection is established,
        # so setup can consume part of the authored 51 ms interval. The real
        # process proof must still show paced dispatch rather than a burst.
        assert 15_000_000 <= start_delta_ns <= 250_000_000
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_config_v2_controls_native_connection_reuse(tmp_path: Path) -> None:
    _ConnectionHandler.peer_ports.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ConnectionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        default_binary = Path(__file__).parents[2] / "target/debug/aiperf runner"
        binary = Path(os.environ.get("AIPERF_RUNNER_BIN", default_binary))
        observed: dict[str, list[bool]] = {}
        peer_ports: dict[str, list[int]] = {}
        for strategy in ("pooled", "never"):
            artifact_dir = tmp_path / strategy
            envelope = AIPerfConfig.model_validate(
                {
                    "benchmark": {
                        "models": ["mock-model"],
                        "endpoint": {
                            "urls": [f"http://127.0.0.1:{port}/v1/chat/completions"],
                            "streaming": False,
                            "use_server_token_count": True,
                            "connection_reuse": strategy,
                        },
                        "dataset": {
                            "type": "synthetic",
                            "entries": 3,
                            "isl": 8,
                            "osl": 1,
                        },
                        "profiling": {
                            "type": "concurrency",
                            "requests": 3,
                            "concurrency": 1,
                        },
                        "artifacts": {"dir": str(artifact_dir), "trace": True},
                        "gpu_telemetry": {"enabled": False},
                        "server_metrics": {"enabled": False},
                        "runtime": {"ui": "none"},
                    }
                }
            )
            run = BenchmarkRun(
                benchmark_id=f"connection-{strategy}",
                cfg=envelope.benchmark,
                artifact_dir=artifact_dir,
                label=strategy,
                random_seed=41,
            )
            start = len(_ConnectionHandler.peer_ports)

            result = NativeExecutor(artifact_dir, binary=binary).execute_sync(
                run
            )

            assert result.success, result.error
            rows = [
                orjson.loads(line)
                for line in (artifact_dir / "profile_export.jsonl")
                .read_bytes()
                .splitlines()
            ]
            observed[strategy] = [
                row["trace_data"]["connection_reused"] for row in rows
            ]
            peer_ports[strategy] = _ConnectionHandler.peer_ports[start:]

        assert observed["pooled"] == [False, True, True]
        assert len(set(peer_ports["pooled"])) == 1
        assert observed["never"] == [False, False, False]
        assert len(set(peer_ports["never"])) == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_config_v2_enforces_one_native_end_to_end_request_timeout(
    tmp_path: Path,
) -> None:
    _AdaptiveChatHandler.active = 0
    _AdaptiveChatHandler.peak = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AdaptiveChatHandler)
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
                        "timeout": 0.01,
                    },
                    "dataset": {
                        "type": "synthetic",
                        "entries": 1,
                        "isl": 8,
                        "osl": 1,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "requests": 1,
                        "concurrency": 1,
                    },
                    "artifacts": {"dir": str(tmp_path), "raw": True},
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        run = BenchmarkRun(
            benchmark_id="native-total-timeout",
            cfg=envelope.benchmark,
            artifact_dir=tmp_path,
            label="timeout",
            random_seed=43,
        )
        default_binary = Path(__file__).parents[2] / "target/debug/aiperf runner"
        binary = Path(os.environ.get("AIPERF_RUNNER_BIN", default_binary))

        result = NativeExecutor(tmp_path, binary=binary).execute_sync(run)

        assert not result.success
        assert result.error == "All 1 requests failed"
        rows = [
            orjson.loads(line)
            for line in (tmp_path / "profile_export_raw.jsonl")
            .read_bytes()
            .splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["responses"] == []
        assert rows[0]["error"] == {
            "code": None,
            "type": "TimeoutError",
            "message": "request timeout after 10000000ns",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_config_v2_adaptive_phase_controls_the_native_live_issuer(
    tmp_path: Path,
) -> None:
    _AdaptiveChatHandler.active = 0
    _AdaptiveChatHandler.peak = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AdaptiveChatHandler)
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
                        "entries": 2,
                        "isl": 8,
                        "osl": 1,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "duration": 8,
                        "concurrency": 2,
                        "adaptive_scale": {
                            "enabled": True,
                            "control": {
                                "variable": "concurrency",
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
                    "artifacts": {"dir": str(tmp_path)},
                    "gpu_telemetry": {"enabled": False},
                    "server_metrics": {"enabled": False},
                    "runtime": {"ui": "none"},
                }
            }
        )
        run = BenchmarkRun(
            benchmark_id="python-rust-adaptive-e2e",
            cfg=envelope.benchmark,
            artifact_dir=tmp_path,
            label="native-adaptive",
            random_seed=37,
        )
        default_binary = Path(__file__).parents[2] / "target/debug/aiperf runner"
        binary = Path(os.environ.get("AIPERF_RUNNER_BIN", default_binary))
        result = NativeExecutor(tmp_path, binary=binary).execute_sync(run)

        assert result.success, result.error
        assert result.summary_metrics["request_count"].avg > 10
        assert _AdaptiveChatHandler.peak >= 2
        summary = orjson.loads((tmp_path / "adaptive_scale_summary.json").read_bytes())
        assert summary["schema_version"] == 2
        assert summary["status"] == "incomplete"
        assert summary["control_variable"] == "concurrency"
        assert summary["control_value"] == 2.0
        assert (
            summary["completed_reason"]
            == "max_control_value_reached_without_saturation"
        )
        events = [
            orjson.loads(line)
            for line in (tmp_path / "adaptive_scale_events.jsonl")
            .read_bytes()
            .splitlines()
        ]
        assert events[0]["event"] == "adaptive_phase_started"
        assert any(
            event["event"] == "adaptive_decision"
            and event["control_value_after"] == 2.0
            for event in events
        )
        assert events[-1]["event"] == "adaptive_incomplete"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
