# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial integration tests for endpoint shape and path overrides.

Focuses on:
- endpoint path overrides whose plugin default path differs from the mock route
- endpoint-specific request payload shapes that are easy to flatten incorrectly
- raw payload replay to a full URL without endpoint path appending

Out of scope:
- endpoint parser unit-level malformed-response handling; see tests/unit/endpoints/
- broad happy-path endpoint coverage; see sibling test_*_endpoint.py files
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults

# ============================================================================
# Helpers
# ============================================================================

_FAST_REQUEST_COUNT = 1
_RAW_CHAT_MODEL = "gpt2"


def _write_raw_chat_payload(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": _RAW_CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Summarize the raw endpoint path override contract.",
            }
        ],
        "stream": False,
        "max_tokens": 6,
    }
    path.write_bytes(orjson.dumps(payload) + b"\n")
    return payload


def _write_named_text_rankings_dataset(path: Path) -> None:
    records = [
        {
            "texts": [
                {
                    "name": "query",
                    "contents": [
                        "Which retrieval passage discusses endpoint payloads?"
                    ],
                },
                {
                    "name": "passages",
                    "contents": [
                        "Endpoint payload formatters preserve request shape.",
                        "GPU telemetry exports Prometheus samples.",
                    ],
                },
            ]
        }
    ]
    path.write_bytes(b"".join(orjson.dumps(record) + b"\n" for record in records))


# ============================================================================
# Endpoint path overrides
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestEndpointPathOverrides:
    """Route endpoints to mock-server paths that are not the plugin defaults."""

    async def test_chat_embeddings_endpoint_override_uses_chat_shape_and_no_streaming_metrics(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
    ) -> None:
        result = await cli.run(
            f"""
            aiperf profile \
                --model nomic-ai/nomic-embed-text-v1.5 \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat_embeddings \
                --endpoint /v1/chat/embeddings \
                --request-count {_FAST_REQUEST_COUNT} \
                --concurrency 1 \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui none
            """,
            timeout=120.0,
        )

        assert result.request_count == _FAST_REQUEST_COUNT
        assert result.json is not None
        assert result.json.time_to_first_token is None
        assert result.has_streaming_metrics is False
        assert result.raw_records is not None
        assert result.raw_records[0].payload.get("messages")
        assert "input" not in result.raw_records[0].payload

    async def test_image_retrieval_endpoint_override_preserves_image_metrics(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
    ) -> None:
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/page-elements-v2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type image_retrieval \
                --endpoint /v1/image/infer \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --request-count {_FAST_REQUEST_COUNT} \
                --concurrency 1 \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui none
            """,
            timeout=120.0,
        )

        assert result.request_count == _FAST_REQUEST_COUNT
        assert result.json is not None
        assert result.json.image_throughput is not None
        assert result.json.image_latency is not None
        assert result.raw_records is not None
        payload = result.raw_records[0].payload
        assert isinstance(payload.get("input"), list)
        assert payload["input"][0]["type"] == "image_url"


# ============================================================================
# Raw and ranking payload shape traps
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestEndpointPayloadShapeTraps:
    """Exercise endpoint types whose accepted payload keys differ by API family."""

    async def test_raw_endpoint_full_chat_url_replays_top_level_messages(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        input_file = tmp_path / "raw-chat-payloads.jsonl"
        payload = _write_raw_chat_payload(input_file)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {_RAW_CHAT_MODEL} \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url}/v1/chat/completions \
                --endpoint-type raw \
                --custom-dataset-type raw_payload \
                --input-file {input_file} \
                --request-count {_FAST_REQUEST_COUNT} \
                --concurrency 1 \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui none
            """,
            timeout=120.0,
        )

        assert result.request_count == _FAST_REQUEST_COUNT
        assert result.raw_records is not None
        assert result.raw_records[0].payload == payload

    async def test_hf_tei_rankings_named_text_objects_build_rerank_payload(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        input_file = tmp_path / "named-text-rankings.jsonl"
        _write_named_text_rankings_dataset(input_file)

        result = await cli.run(
            f"""
            aiperf profile \
                --model BAAI/bge-reranker-large \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type hf_tei_rankings \
                --custom-dataset-type single_turn \
                --input-file {input_file} \
                --request-count {_FAST_REQUEST_COUNT} \
                --concurrency 1 \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui none
            """,
            timeout=120.0,
        )

        assert result.request_count == _FAST_REQUEST_COUNT
        assert result.raw_records is not None
        payload = result.raw_records[0].payload
        assert (
            payload["query"] == "Which retrieval passage discusses endpoint payloads?"
        )
        assert payload["texts"] == [
            "Endpoint payload formatters preserve request shape.",
            "GPU telemetry exports Prometheus samples.",
        ]
        assert "passages" not in payload
