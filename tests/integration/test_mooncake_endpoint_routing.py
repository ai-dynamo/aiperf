# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for per-row endpoint routing in Mooncake traces.

A trace row may name a registered endpoint plugin via ``endpoint_type``,
sending that request to a different endpoint of the same server than the
run-level ``--endpoint-type``. These tests pin the behavior end-to-end: that
each row reaches the URL path its endpoint declares (asserted against the mock
server's own per-request recording, not against AIPerf's view of itself), and
that requests to a non-primary endpoint stay out of the primary metric set.
"""

from collections import Counter
from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.utils import create_mooncake_trace_file

CHAT_PATH = "/v1/chat/completions"
EMBEDDINGS_PATH = "/v1/embeddings"


def _rag_trace_rows() -> list[dict]:
    """A RAG-shaped Mooncake trace: each query is embedded, then answered.

    Every row carries the native Mooncake fields (``timestamp``,
    ``input_length``, ``hash_ids``); the chat rows share the leading blocks
    ``[1, 2]`` so the run exercises prefix reuse the way a real trace would.
    Timestamps are compressed to keep the fixed-schedule timeline inside the
    integration timeout while preserving the interleaving that makes routing
    meaningful -- an embedding and a chat request are in flight against the
    same server at overlapping times.

    ``input_length`` and ``hash_ids`` must satisfy the block-size contract in
    ``coding_content._build_token_sequence``: with the default 512-token
    block, a row with ``m`` hashes needs
    ``0 < input_length - (m - 1) * 512 <= 512`` (or an input long enough that
    every hash is a full block). Rows that violate it fail dataset
    configuration outright, so keep the arithmetic in mind when editing.

    ``output_length`` is omitted on the embedding rows because that endpoint
    generates no tokens; ``test_native_shaped_trace_ignores_output_length``
    covers the native shape where every row carries it.
    """
    return [
        {"timestamp": 0,   "input_length": 64,   "hash_ids": [10],         "endpoint_type": "embeddings"},
        {"timestamp": 20,  "input_length": 1100, "output_length": 24, "hash_ids": [1, 2, 3]},
        {"timestamp": 60,  "input_length": 48,   "hash_ids": [11],         "endpoint_type": "embeddings"},
        {"timestamp": 80,  "input_length": 1180, "output_length": 32, "hash_ids": [1, 2, 4]},
        {"timestamp": 140, "input_length": 72,   "hash_ids": [12],         "endpoint_type": "embeddings"},
        {"timestamp": 160, "input_length": 1050, "output_length": 16, "hash_ids": [1, 2, 5]},
    ]  # fmt: skip


def _recorded_endpoints(record_path: Path) -> Counter:
    """Return the count of requests the mock server received per URL path."""
    with open(record_path, "rb") as f:
        return Counter(orjson.loads(line)["endpoint"] for line in f if line.strip())


@pytest.mark.integration
@pytest.mark.asyncio
class TestMooncakeEndpointRouting:
    """End-to-end per-row endpoint routing against the mock server."""

    async def test_rows_reach_their_own_endpoint_path(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        tmp_path: Path,
    ) -> None:
        """Routed rows hit /v1/embeddings; the rest hit the run-level path.

        Asserted from the server side (``--record-requests``) so the test
        cannot pass on AIPerf merely believing it routed correctly.
        """
        rows = _rag_trace_rows()
        trace_file = create_mooncake_trace_file(tmp_path, rows)
        record_path = tmp_path / "received.jsonl"

        async with mock_server_factory(
            record_requests=str(record_path),
            tokenizer="builtin",
            no_tokenizer=False,
            fast=True,
        ) as server:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model {defaults.model} \
                    --url {server.url} \
                    --endpoint-type chat \
                    --input-file {trace_file} \
                    --custom-dataset-type mooncake_trace \
                    --request-count {len(rows)} \
                    --fixed-schedule \
                    --fixed-schedule-auto-offset \
                    --workers-max {defaults.workers_max} \
                    --ui {defaults.ui}
                """
            )

        assert result.exit_code == 0

        expected_embeddings = sum(1 for r in rows if "endpoint_type" in r)
        expected_chat = len(rows) - expected_embeddings
        assert _recorded_endpoints(record_path) == Counter(
            {CHAT_PATH: expected_chat, EMBEDDINGS_PATH: expected_embeddings}
        )

    async def test_routed_requests_excluded_from_metrics(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        tmp_path: Path,
    ) -> None:
        """Metrics describe the run-level endpoint only.

        An embeddings response has no output tokens, so folding those records
        into the chat metrics would corrupt TTFT/ITL/OSL. They are reported as
        a separate count under ``metadata.unmeasured_request_counts`` instead.
        """
        rows = _rag_trace_rows()
        trace_file = create_mooncake_trace_file(tmp_path, rows)

        async with mock_server_factory(fast=True) as server:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model {defaults.model} \
                    --url {server.url} \
                    --endpoint-type chat \
                    --input-file {trace_file} \
                    --custom-dataset-type mooncake_trace \
                    --request-count {len(rows)} \
                    --fixed-schedule \
                    --fixed-schedule-auto-offset \
                    --workers-max {defaults.workers_max} \
                    --ui {defaults.ui}
                """
            )

        assert result.exit_code == 0

        routed = sum(1 for r in rows if "endpoint_type" in r)
        assert result.request_count == len(rows) - routed, (
            "request_count must cover the run-level endpoint only"
        )
        assert result.json is not None
        metadata = getattr(result.json, "metadata", None)
        assert metadata is not None, "export carries no run metadata"
        assert metadata["unmeasured_request_counts"] == {"embeddings": routed}

    async def test_native_shaped_trace_ignores_output_length(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        tmp_path: Path,
    ) -> None:
        """A native-shaped trace routes without edits.

        Published Mooncake traces carry ``output_length`` on every row. Routing
        such a row to an endpoint that generates no tokens must not fail or
        require stripping the field -- the value is ignored (reported once at
        load time) and the request still goes out.
        """
        rows = [
            {"timestamp": 0,  "input_length": 600, "output_length": 24, "hash_ids": [1, 2]},
            {"timestamp": 40, "input_length": 32,  "output_length": 40, "hash_ids": [3], "endpoint_type": "embeddings"},
        ]  # fmt: skip
        trace_file = create_mooncake_trace_file(tmp_path, rows)
        record_path = tmp_path / "received.jsonl"

        async with mock_server_factory(
            record_requests=str(record_path),
            tokenizer="builtin",
            no_tokenizer=False,
            fast=True,
        ) as server:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model {defaults.model} \
                    --url {server.url} \
                    --endpoint-type chat \
                    --input-file {trace_file} \
                    --custom-dataset-type mooncake_trace \
                    --request-count {len(rows)} \
                    --fixed-schedule \
                    --fixed-schedule-auto-offset \
                    --workers-max {defaults.workers_max} \
                    --ui {defaults.ui}
                """
            )

        assert result.exit_code == 0
        assert _recorded_endpoints(record_path) == Counter(
            {CHAT_PATH: 1, EMBEDDINGS_PATH: 1}
        )
