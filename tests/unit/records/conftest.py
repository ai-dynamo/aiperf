# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for records tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import (
    ErrorDetails,
    RequestInfo,
    RequestRecord,
    SSEMessage,
    Text,
    TextResponse,
    Turn,
)
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.plugin.enums import EndpointType
from aiperf.records.inference_result_parser import InferenceResultParser


def create_test_request_info(
    model_name: str = "test-model",
    conversation_id: str = "cid",
    turn_index: int = 0,
    turns: list[Turn] | None = None,
) -> RequestInfo:
    """Create a RequestInfo for testing."""
    return RequestInfo(
        model_endpoint=ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name=model_name)],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000/v1/test",
            ),
        ),
        turns=turns or [],
        turn_index=turn_index,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="test-request-id",
        x_correlation_id="test-correlation-id",
        conversation_id=conversation_id,
    )


@pytest.fixture
def sample_turn():
    """Sample turn with 4 text strings (8 words total) for testing."""
    return Turn(
        role="user",
        texts=[
            Text(contents=["Hello world", " Test case"]),
            Text(contents=["Another input", " Final message"]),
        ],
    )


@pytest.fixture
def inference_result_parser(user_config):
    """Create an InferenceResultParser with mocked dependencies."""

    def mock_communication_init(self, service_config, **kwargs):
        from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin

        AIPerfLifecycleMixin.__init__(self, service_config=service_config, **kwargs)
        self.service_config = service_config
        self.comms = MagicMock()
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
        patch(
            "aiperf.common.models.model_endpoint_info.ModelEndpointInfo.from_user_config"
        ),
        patch("aiperf.plugin.plugins.get_class"),
        patch("aiperf.plugin.plugins.get_endpoint_metadata"),
    ):
        parser = InferenceResultParser(
            service_config=ServiceConfig(),
            user_config=user_config,
        )
        return parser


@pytest.fixture
def setup_inference_parser(inference_result_parser, mock_tokenizer_cls):
    """Setup InferenceResultParser for testing with mocked tokenizer."""
    tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
    inference_result_parser.get_tokenizer = AsyncMock(return_value=tokenizer)
    inference_result_parser.endpoint = MagicMock()
    return inference_result_parser


def create_invalid_record(
    *,
    no_responses: bool = False,
    bad_start_timestamp: bool = False,
    bad_response_timestamps: list[int] | None = None,
    has_error: bool = False,
    no_content_responses: bool = False,
    model_name: str = "test-model",
    turns: list[Turn] | None = None,
) -> RequestRecord:
    """Create an invalid RequestRecord for testing.

    Args:
        no_responses: If True, creates a record with no responses
        bad_start_timestamp: If True, sets start_perf_ns to -1
        bad_response_timestamps: List of invalid perf_ns values for responses
        has_error: If True, adds an existing error to the record
        no_content_responses: If True, creates responses without content (e.g., [DONE] markers)
        model_name: Model name for the record
        turns: Optional list of turns to include in the record

    Returns:
        RequestRecord with the specified invalid configuration
    """
    record = RequestRecord(
        request_info=create_test_request_info(model_name=model_name, turns=turns),
        model_name=model_name,
        turns=turns or [],
    )

    if has_error:
        record.error = ErrorDetails(
            code=500, message="Original error", type="ServerError"
        )

    if bad_start_timestamp:
        record.start_perf_ns = -1

    if no_responses:
        record.responses = []
    elif no_content_responses:
        # Create responses with valid timestamps but no actual content
        record.responses = [
            SSEMessage.parse("[DONE]", perf_ns=1000),
            TextResponse(perf_ns=2000, content_type="text/plain", text=""),
        ]
    elif bad_response_timestamps:
        record.responses = [
            TextResponse(
                perf_ns=perf_ns, content_type="text/plain", text=f"response {i}"
            )
            for i, perf_ns in enumerate(bad_response_timestamps)
        ]

    return record


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns token count based on word count."""
    tokenizer = MagicMock(spec=Tokenizer)
    tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
    return tokenizer
