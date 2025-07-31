# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.messages import ConversationTurnResponseMessage
from aiperf.common.models import RequestRecord, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.inference.parsers.inference_result_parser import (
    InferenceResultParser,
)


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=Tokenizer)
    tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
    return tokenizer


@pytest.fixture
def sample_turn():
    return Turn(
        role="user",
        texts=[
            Text(contents=["Hello world", "Test case"]),
            Text(contents=["Another input", "Final message"]),
        ],
    )


@pytest.fixture
def mock_turn_response(sample_turn):
    return ConversationTurnResponseMessage(
        service_id="test-service",
        conversation_id="cid",
        turn_index=0,
        turn=sample_turn,
    )


@pytest.fixture
def sample_request_record():
    return RequestRecord(conversation_id="cid", turn_index=0)


@pytest.fixture
def parser(mock_turn_response):
    with patch.object(InferenceResultParser, "__init__", lambda self: None):
        parser = InferenceResultParser()
        parser.service_id = "test-parser"
        parser.conversation_request_client = MagicMock()
        parser.conversation_request_client.request = AsyncMock(
            return_value=mock_turn_response
        )
        return parser


@pytest.mark.asyncio
async def test_compute_input_token_count(parser, sample_request_record, mock_tokenizer):
    result = await parser.compute_input_token_count(
        sample_request_record, mock_tokenizer
    )

    assert result == 8  # 4 strings × 2 words each
    assert mock_tokenizer.encode.call_count == 4
