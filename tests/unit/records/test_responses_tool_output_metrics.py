# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pytest import param

from aiperf.common.models import RequestRecord, TextResponse, Turn
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from aiperf.records.inference_result_parser import InferenceResultParser
from tests.unit.endpoints.conftest import create_model_endpoint, create_request_info


@pytest.mark.asyncio
@pytest.mark.parametrize("apply_chat_template", [False, True])
@pytest.mark.parametrize(
    "output,tool_tokens,image_count",
    [
        param("tool result", 2, 0, id="string-output"),
        param([{"type": "input_text", "text": "tool result"}], 2, 0, id="text-parts"),
        param([
            {"type": "input_text", "text": "first result"},
            {"type": "input_image", "image_url": "data:image/png;base64,a"},
            {"type": "input_text", "text": "second result"},
            {"type": "input_image", "file_id": "file-image"},
        ], 4, 2, id="text-and-images"),
        param([{"type": "input_image", "image_url": "data:image/png;base64,a"}], 0, 1, id="image-only"),
        param([], 0, 0, id="empty-output"),
    ],
)  # fmt: skip
async def test_responses_function_output_counts_wire_text_and_images(
    *,
    setup_inference_parser: InferenceResultParser,
    apply_chat_template: bool,
    output: str | list[dict[str, Any]],
    tool_tokens: int,
    image_count: int,
) -> None:
    model_endpoint = create_model_endpoint(EndpointType.RESPONSES)
    endpoint = ResponsesEndpoint(model_endpoint=model_endpoint)
    info = create_request_info(
        model_endpoint=model_endpoint,
        turns=[
            Turn(
                raw_messages=[
                    {"role": "user", "content": "read result"},
                    {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": output,
                    },
                ]
            )
        ],
    )
    payload = endpoint.format_payload(info)
    assert payload["input"][1]["output"] == output
    info.payload_bytes = orjson.dumps(payload)
    record = RequestRecord(
        request_info=info,
        model_name="test-model",
        start_perf_ns=1,
        responses=[
            TextResponse(
                perf_ns=2,
                text=orjson.dumps(
                    {
                        "object": "response",
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "done"}],
                            }
                        ],
                    }
                ).decode(),
            )
        ],
    )
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
    tokenizer._tokenizer.apply_chat_template.return_value = list(range(7))
    setup_inference_parser.endpoint = endpoint
    setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)
    setup_inference_parser.run.cfg.tokenizer.apply_chat_template = apply_chat_template

    parsed = await setup_inference_parser.parse_request_record(record)

    assert record.error is None
    assert parsed.token_counts.input == (7 if apply_chat_template else 2) + tool_tokens
    assert parsed.media_counts.images == image_count
    assert parsed.media_counts.audios == 0
    assert parsed.media_counts.videos == 0
