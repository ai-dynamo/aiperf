# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A remote --input-file replays exactly like the same file on local disk.

Serves a Mooncake trace from a throwaway HTTP server (Last-Modified + Content-Length
version, no Range support), so the real https resolver, preflight, cache, and replay
path all run with no mocks.
"""

import functools
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import orjson
import pytest

from aiperf.common.environment import Environment
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI

_ROWS = [
    {"input_length": 48, "output_length": 8, "hash_ids": [1, 2, 3], "timestamp": 100},
    {"input_length": 48, "output_length": 8, "hash_ids": [4, 5, 6], "timestamp": 200},
    {"input_length": 48, "output_length": 8, "hash_ids": [7, 8, 9], "timestamp": 300},
]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        """Keep the throwaway server out of test output."""


@pytest.fixture
def trace_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    root = tmp_path / "srv"
    root.mkdir()
    (root / "mooncake.jsonl").write_bytes(
        b"\n".join(orjson.dumps(r) for r in _ROWS) + b"\n"
    )
    cache = tmp_path / "remote-cache"
    monkeypatch.setenv("AIPERF_DATASET_REMOTE_CACHE_DIR", str(cache))
    monkeypatch.setattr(Environment.DATASET, "REMOTE_CACHE_DIR", cache)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    handler = functools.partial(_QuietHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/mooncake.jsonl"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.component_integration
class TestRemoteInputFile:
    def test_timestamped_remote_trace_auto_promotes_and_replays_every_row(
        self, cli: AIPerfCLI, trace_url: str
    ) -> None:
        """No --fixed-schedule and no --request-count: the remote trace is promoted
        from its first record, exactly as the same local file would be."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --custom-dataset-type mooncake_trace \
                --input-file {trace_url} \
                --isl-block-size 16 \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui {defaults.ui}
            """,
            timeout=90.0,
        )
        assert result.raw_records is not None
        assert len(result.raw_records) == len(_ROWS)

    def test_no_fixed_schedule_respects_request_count(
        self, cli: AIPerfCLI, trace_url: str
    ) -> None:
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --custom-dataset-type mooncake_trace \
                --input-file {trace_url} \
                --isl-block-size 16 \
                --no-fixed-schedule \
                --request-count 2 \
                --concurrency 1 \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui {defaults.ui}
            """,
            timeout=90.0,
        )
        assert result.raw_records is not None
        assert len(result.raw_records) == 2
