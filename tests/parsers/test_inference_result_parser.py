# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import EndpointConfig, InputConfig, ServiceConfig, UserConfig
from aiperf.common.messages import ConversationTurnResponseMessage
from aiperf.common.models import ErrorDetails, RequestRecord, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.parsers.inference_result_parser import InferenceResultParser


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns token count based on word count."""
    tokenizer = MagicMock(spec=Tokenizer)
    tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
    return tokenizer


@pytest.fixture
def sample_turn():
    """Sample turn with 4 text strings (8 words total)."""
    return Turn(
        role="user",
        texts=[
            Text(contents=["Hello world", " Test case"]),
            Text(contents=["Another input", " Final message"]),
        ],
    )


@pytest.fixture
def mock_turn_response(sample_turn):
    return ConversationTurnResponseMessage(
        service_id="test-service",
        turn=sample_turn,
    )


@pytest.fixture
def parser(mock_turn_response):
    """Create a parser with mocked communications layer."""
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=mock_turn_response)

    mock_comms = MagicMock()
    mock_comms.create_request_client.return_value = mock_client

    def mock_communication_init(self, **_kwargs):
        self.comms = mock_comms
        # Add logger methods
        for method in [
            "trace_or_debug",
            "debug",
            "info",
            "warning",
            "error",
            "exception",
        ]:
            setattr(self, method, MagicMock())

    with (
        patch(
            "aiperf.common.mixins.CommunicationMixin.__init__", mock_communication_init
        ),
        patch("aiperf.clients.model_endpoint_info.ModelEndpointInfo.from_user_config"),
        patch("aiperf.common.factories.ResponseExtractorFactory.create_instance"),
    ):
        parser = InferenceResultParser(
            service_config=ServiceConfig(),
            user_config=UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(),
            ),
        )
        return parser


def create_request_record(has_error=False, is_invalid=False, model_name="test-model"):
    """Helper to create request records with various states."""
    record = RequestRecord(conversation_id="cid", turn_index=0, model_name=model_name)

    if has_error:
        record.error = ErrorDetails(
            code=500, message="Server error", type="ServerError"
        )
    if is_invalid:
        record._valid = False

    return record


def setup_parser_for_error_tests(parser, mock_tokenizer, sample_turn):
    """Common setup for error record tests."""
    parser.get_tokenizer = AsyncMock(return_value=mock_tokenizer)
    parser.get_turn = AsyncMock(return_value=sample_turn)
    parser.extractor = MagicMock()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record_type",
    ["error", "invalid", "processing_exception"],
)
async def test_error_records_compute_input_tokens(
    parser, mock_tokenizer, sample_turn, record_type
):
    """Test that input_token_count is computed for all error scenarios."""
    if record_type == "error":
        record = create_request_record(has_error=True)
    elif record_type == "invalid":
        record = create_request_record(is_invalid=True)
    else:  # processing_exception
        record = create_request_record()

    setup_parser_for_error_tests(parser, mock_tokenizer, sample_turn)

    if record_type == "processing_exception":
        parser.extractor.extract_response_data = AsyncMock(
            side_effect=ValueError("Processing failed")
        )

    result = await parser.parse_request_record(record)

    assert result.request == record
    assert result.input_token_count == 8
    assert result.responses == []
    assert record.error is not None
