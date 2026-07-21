# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: ISL and media counts must come from ``payload_bytes``, not ``turns``.

On the worker payload-bytes fast path (the common mmap path) the record's
``turns`` are a content-free stub ``[Turn(role="user")]`` -- the canonical body
is ``request_info.payload_bytes``. Before this fix the parser re-derived ISL
from ``turns`` and ``NumImagesMetric`` summed ``turn.images``, so on the fast
path both were silently wrong (ISL ~0, image_count 0) even though the wire
payload carried real text and images.

This test builds a record with a fully-populated chat ``payload_bytes`` but
EMPTY ``turns`` and asserts the parser still recovers the correct input token
count and image count. With the pre-fix turns-based parser this fails
(token_counts.input is None, media_counts.images is 0).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.models import ParsedResponse, RequestRecord
from aiperf.common.models.record_models import TextResponseData
from tests.unit.records.conftest import create_test_request_info


@pytest.mark.asyncio
async def test_isl_and_image_count_from_payload_bytes_with_empty_turns(
    setup_inference_parser, mock_tokenizer
) -> None:
    # Word-counting tokenizer: token count == number of whitespace-split words.
    setup_inference_parser.get_tokenizer = AsyncMock(return_value=mock_tokenizer)
    setup_inference_parser.endpoint.extract_response_data = MagicMock(
        return_value=[ParsedResponse(perf_ns=1500, data=TextResponseData(text="ok"))]
    )

    # Canonical wire body: one user message with five words of text + 2 images.
    payload = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "one two three four five"},
                    {"type": "image_url", "image_url": {"url": "data:img-0"}},
                    {"type": "image_url", "image_url": {"url": "data:img-1"}},
                ],
            }
        ],
    }

    # Fast-path shape: payload_bytes is canonical; turns is an empty stub.
    request_info = create_test_request_info(turns=[])
    request_info.payload_bytes = orjson.dumps(payload)
    record = RequestRecord(
        model_name="test-model",
        request_info=request_info,
        turns=[],
        start_perf_ns=1000,
        timestamp_ns=1000,
        end_perf_ns=2000,
        status=200,
        responses=[],
    )

    parsed = await setup_inference_parser.parse_request_record(record)

    # ISL recovered from payload_bytes text (5 words) -- NOT from the empty turns.
    assert parsed.token_counts is not None
    assert parsed.token_counts.input == 5
    # Image count recovered from payload_bytes -- NOT from the empty turns.
    assert parsed.media_counts.images == 2
