# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the mock server's per-request ISL/OSL recorder mode.

The recorder lives in the FastAPI lifespan: each request is tokenized inline
with the configured tokenizer and written as one JSONL line, and a sibling
.summary.json file is emitted on server shutdown.
"""

from pathlib import Path

import aiohttp
import orjson
import pytest


@pytest.mark.integration
@pytest.mark.asyncio
class TestRecordRequests:
    async def test_records_per_request_isl_and_requested_osl(
        self, tmp_path: Path, mock_server_factory
    ) -> None:
        record_path = tmp_path / "requests.jsonl"
        summary_path = tmp_path / "requests.jsonl.summary.json"

        async with (
            mock_server_factory(
                record_requests=str(record_path),
                tokenizer="builtin",
                no_tokenizer=False,
                fast=True,
            ) as server,
            aiohttp.ClientSession() as http,
        ):
            for max_tokens in (16, 32, 64, 128, 256):
                async with http.post(
                    f"{server.url}/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [
                            {"role": "user", "content": "Hello, world! " * 20}
                        ],
                        "max_tokens": max_tokens,
                    },
                ) as resp:
                    assert resp.status == 200, await resp.text()

            # A chat request that uses the modern max_completion_tokens field
            # plus min_tokens and reasoning_effort so we can verify each is
            # recorded raw.
            async with http.post(
                f"{server.url}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "think for a sec"}],
                    "max_completion_tokens": 192,
                    "min_tokens": 32,
                    "ignore_eos": True,
                    "reasoning_effort": "high",
                },
            ) as resp:
                assert resp.status == 200, await resp.text()

            async with http.post(
                f"{server.url}/v1/completions",
                json={
                    "model": "test",
                    "prompt": "The quick brown fox " * 10,
                    "max_tokens": 48,
                },
            ) as resp:
                assert resp.status == 200, await resp.text()

            async with http.post(
                f"{server.url}/v1/embeddings",
                json={
                    "model": "test",
                    "input": "embed me please",
                },
            ) as resp:
                assert resp.status == 200, await resp.text()

        assert record_path.exists(), "recorder did not write the JSONL file"
        assert summary_path.exists(), "recorder did not write the summary"

        lines = [
            orjson.loads(line) for line in record_path.read_bytes().splitlines() if line
        ]
        assert len(lines) == 8

        chat = [r for r in lines if r["endpoint"] == "/v1/chat/completions"]
        assert len(chat) == 6
        assert all(r["isl"] > 0 for r in chat)
        assert all(r["request_id"].startswith("chatcmpl-") for r in chat)

        # The five legacy max_tokens requests should record max_tokens raw and
        # leave max_completion_tokens null; requested_osl is the resolved cap.
        legacy = [r for r in chat if r["max_completion_tokens"] is None]
        assert {r["max_tokens"] for r in legacy} == {16, 32, 64, 128, 256}
        assert {r["requested_osl"] for r in legacy} == {16, 32, 64, 128, 256}
        assert all(r["reasoning_effort"] is None for r in legacy)
        assert all(r["ignore_eos"] is False for r in legacy)
        assert all(r["min_tokens"] is None for r in legacy)

        # The modern request should preserve the raw field names.
        modern = [r for r in chat if r["max_completion_tokens"] is not None][0]
        assert modern["max_tokens"] is None
        assert modern["max_completion_tokens"] == 192
        assert modern["requested_osl"] == 192
        assert modern["min_tokens"] == 32
        assert modern["ignore_eos"] is True
        assert modern["reasoning_effort"] == "high"

        cmpl = [r for r in lines if r["endpoint"] == "/v1/completions"]
        assert len(cmpl) == 1
        assert cmpl[0]["max_tokens"] == 48
        assert cmpl[0]["max_completion_tokens"] is None
        assert cmpl[0]["requested_osl"] == 48
        assert cmpl[0]["isl"] > 0

        emb = [r for r in lines if r["endpoint"] == "/v1/embeddings"]
        assert len(emb) == 1
        assert emb[0]["requested_osl"] is None
        assert emb[0]["max_tokens"] is None
        assert emb[0]["max_completion_tokens"] is None
        assert emb[0]["isl"] > 0

        summary = orjson.loads(summary_path.read_bytes())
        assert summary["total_requests"] == 8

        chat_stats = summary["per_endpoint"]["/v1/chat/completions"]
        assert chat_stats["count"] == 6
        assert chat_stats["isl"]["mean"] > 0
        assert chat_stats["requested_osl"]["min"] == 16.0
        assert chat_stats["requested_osl"]["max"] == 256.0
        assert chat_stats["min_tokens"]["min"] == 32.0
        assert chat_stats["ignore_eos_count"] == 1
        assert chat_stats["reasoning_effort_counts"] == {"high": 1}

        emb_stats = summary["per_endpoint"]["/v1/embeddings"]
        assert emb_stats["count"] == 1
        assert emb_stats["requested_osl"] is None
        assert emb_stats["min_tokens"] is None
        assert emb_stats["ignore_eos_count"] == 0
        assert emb_stats["reasoning_effort_counts"] is None

        # Histogram + unique_values on the chat ISL block.
        assert chat_stats["isl"]["histogram"] is not None
        chat_isl_hist = chat_stats["isl"]["histogram"]
        assert len(chat_isl_hist["bin_edges"]) == len(chat_isl_hist["counts"]) + 1
        assert sum(chat_isl_hist["counts"]) == chat_stats["count"]
        assert chat_stats["isl"]["unique_values"] >= 1

        # Resolved requested_osl spans six distinct values across the chat fixture:
        # max_tokens in {16, 32, 64, 128, 256} on five requests plus
        # max_completion_tokens=192 on the sixth.
        assert chat_stats["requested_osl"]["unique_values"] == 6
        chat_osl_hist = chat_stats["requested_osl"]["histogram"]
        assert chat_osl_hist is not None
        assert sum(chat_osl_hist["counts"]) == chat_stats["count"]

        # Embeddings: ISL still has a histogram and unique_values.
        assert emb_stats["isl"]["histogram"] is not None
        assert emb_stats["isl"]["unique_values"] >= 1

        # Vocab distribution block for chat: present, with valid shape and ids.
        chat_vd = chat_stats["vocab_distribution"]
        assert chat_vd is not None
        assert chat_vd["vocab_size"] > 0
        assert chat_vd["unique_ids"] >= 1
        assert chat_vd["total_tokens"] >= chat_vd["unique_ids"]
        assert 0.0 <= chat_vd["coverage_pct"] <= 100.0
        assert len(chat_vd["shape_80"]) == 80
        assert sum(chat_vd["shape_80"]) == chat_vd["total_tokens"]
        assert chat_vd["shape_80_stats"]["mean"] == pytest.approx(
            chat_vd["total_tokens"] / 80
        )
        assert chat_vd["shape_80_stats"]["p50"] >= 0.0
        assert 1 <= len(chat_vd["top_tokens"]) <= 10
        for entry in chat_vd["top_tokens"]:
            assert isinstance(entry["id"], int)
            assert isinstance(entry["text"], str)
            assert entry["count"] >= 1
        assert 0.0 <= chat_vd["entropy_bits"] <= chat_vd["max_entropy_bits"] + 1e-6
        assert chat_vd["vocab_size_source"] in {"tokenizer", "observed"}
        # Embeddings endpoint exists in the fixture; its vocab block should
        # also exist (ISL is recorded) — sanity-check that this isn't broken.
        emb_vd = emb_stats["vocab_distribution"]
        assert emb_vd is not None
        assert emb_vd["unique_ids"] >= 1

    async def test_record_requests_forces_workers_to_one(self) -> None:
        """The validator must collapse workers to 1 whenever recording is on —
        the recorder keeps per-request stats in-process, so a single uvicorn
        worker is the supported producer."""
        from aiperf_mock_server.config import MockServerConfig

        cfg = MockServerConfig(record_requests="/tmp/anything.jsonl", workers=8)
        assert cfg.workers == 1

    async def test_record_requests_requires_a_tokenizer(self) -> None:
        """Recording counts ISL with the real tokenizer, so disabling the
        tokenizer while requesting recording is incoherent and must fail
        fast — not crash later inside a request handler."""
        from aiperf_mock_server.config import MockServerConfig
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError, match="--record-requests requires a tokenizer"
        ):
            MockServerConfig(
                record_requests="/tmp/anything.jsonl",
                no_tokenizer=True,
            )
