# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration test: ``--per-chunk-usage`` drives the full path end to
end against the mock server -- ``continuous_usage_stats`` is requested, the mock
reports cumulative per-chunk usage, the first content chunk is bundled, and
inter-token latency is emitted from the server-reported first-chunk count.

The mock is told to bundle the first three output tokens into the first streamed
chunk (``mock_first_chunk_tokens:3``), emulating a TRT-LLM ``stream-interval``
server. ``test-model`` avoids the mock's reasoning path so the bundled output
chunk is genuinely the first content chunk.
"""

import math

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI

FIRST_CHUNK_TOKENS = 3


def _first_content_chunk_usage(raw_record) -> int | None:
    """Cumulative completion_tokens on the first streamed chunk carrying content."""
    for response in raw_record.responses:
        chunk = response.get_json()
        if not chunk:
            continue
        choices = chunk.get("choices") or []
        delta = (choices[0].get("delta") if choices else {}) or {}
        if delta.get("content"):
            usage = chunk.get("usage") or {}
            return usage.get("completion_tokens")
    return None


@pytest.mark.component_integration
class TestPerChunkUsageMetric:
    """The --per-chunk-usage path works end to end and feeds ITL the real
    first-content-chunk token count from the server's per-chunk usage."""

    def test_per_chunk_usage_bundled_first_chunk(self, cli: AIPerfCLI) -> None:
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model test-model --tokenizer {defaults.tokenizer} \
                --endpoint-type chat --streaming \
                --per-chunk-usage --use-server-token-count \
                --extra-inputs ignore_eos:true \
                --extra-inputs mock_first_chunk_tokens:{FIRST_CHUNK_TOKENS} \
                --synthetic-input-tokens-mean 50 --output-tokens-mean 20 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui {defaults.ui}
            """,
            timeout=defaults.timeout,
        )
        assert result.request_count == defaults.request_count

        # ITL is emitted and finite/positive.
        itl = getattr(result.json, "inter_token_latency", None)
        assert itl is not None, "inter_token_latency missing from JSON export"
        itl_avg = itl["avg"] if isinstance(itl, dict) else itl.avg
        assert math.isfinite(itl_avg) and itl_avg > 0, (
            f"ITL not finite/positive: {itl_avg}"
        )

        # Deterministic proof the fix's input is correct: continuous_usage_stats
        # was requested and the mock reported the first content chunk carrying the
        # bundled count (3), which is what ITL subtracts.
        assert result.raw_records, "raw records must be exported"
        checked = 0
        for rec in result.raw_records:
            stream_options = (rec.payload or {}).get("stream_options") or {}
            assert stream_options.get("continuous_usage_stats") is True, (
                f"continuous_usage_stats not requested: {stream_options}"
            )
            first_chunk = _first_content_chunk_usage(rec)
            assert first_chunk == FIRST_CHUNK_TOKENS, (
                f"first content chunk usage should be {FIRST_CHUNK_TOKENS}, got {first_chunk}"
            )
            checked += 1
        assert checked > 0, "no raw records inspected"
