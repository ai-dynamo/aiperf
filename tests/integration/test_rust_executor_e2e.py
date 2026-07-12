# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python Config v2 -> stdio -> Rust HTTP/SSE -> native-v2 proof."""

from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

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


class _ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    bodies: list[dict[str, object]] = []

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/dataset/sharegpt.json":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(_SHAREGPT)))
        self.end_headers()
        self.wfile.write(_SHAREGPT)

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
        file_result = RustSubprocessExecutor(
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
        template_result = RustSubprocessExecutor(
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
        multimodal_result = RustSubprocessExecutor(
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
            public_result = RustSubprocessExecutor(
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
        synthesis_result = RustSubprocessExecutor(
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
