# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for usage field passthrough in InferenceResultParser.

The parser should NOT aggregate or modify usage data - it should pass through
the raw responses as-is, letting the metrics layer handle extraction.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.models import (
    ParsedResponse,
    RequestRecord,
    Text,
    TextResponseData,
    Turn,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.records.inference_result_parser import InferenceResultParser
from tests.unit.records.conftest import create_test_request_info


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns token count based on word count."""
    tokenizer = MagicMock(spec=Tokenizer)
    tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
    return tokenizer


@pytest.fixture
def parser():
    """Create a parser with mocked endpoint."""
    mock_endpoint = MagicMock()

    def mock_communication_init(self, run, **kwargs):
        AIPerfLifecycleMixin.__init__(self, **kwargs)
        self.run = run
        for method in [
            "trace_or_debug",
            "debug",
            "info",
            "warning",
            "error",
            "exception",
        ]:
            setattr(self, method, MagicMock())

    config = AIPerfConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            }
        },
        phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
    )

    with (
        patch(
            "aiperf.common.mixins.CommunicationMixin.__init__", mock_communication_init
        ),
        patch(
            "aiperf.plugin.plugins.get_class",
            return_value=mock_endpoint,
        ),
        patch("aiperf.plugin.plugins.get_endpoint_metadata"),
    ):
        run = BenchmarkRun(
            benchmark_id="test", cfg=config, artifact_dir=Path("/tmp/test")
        )
        parser = InferenceResultParser(
            run=run,
        )
        parser.endpoint = mock_endpoint
        return parser


def create_test_record() -> RequestRecord:
    """Helper to create a simple test RequestRecord."""
    return RequestRecord(
        model_name="test-model",
        request_info=create_test_request_info(
            turns=[Turn(role="user", texts=[Text(contents=["Test input"])])]
        ),
    )


class TestUsagePassthrough:
    """Tests verifying parser passes through usage data unchanged."""

    @pytest.mark.asyncio
    async def test_passthrough_single_response(self, parser, mock_tokenizer):
        """Test parser passes through single response usage unchanged."""
        parser.endpoint.extract_response_data.return_value = [
            ParsedResponse(
                perf_ns=100,
                data=TextResponseData(text="Hello"),
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ]
        parser.tokenizers = {"test-model": mock_tokenizer}

        result = await parser.process_valid_record(create_test_record())

        assert len(result.responses) == 1
        assert result.responses[0].usage.prompt_tokens == 10
        assert result.responses[0].usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_passthrough_streaming_cumulative(self, parser, mock_tokenizer):
        """Test parser passes through cumulative streaming usage."""
        parser.endpoint.extract_response_data.return_value = [
            ParsedResponse(
                perf_ns=100,
                data=TextResponseData(text="Hello"),
                usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            ),
            ParsedResponse(
                perf_ns=200,
                data=TextResponseData(text=" world"),
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            ),
            ParsedResponse(
                perf_ns=300,
                data=TextResponseData(text="!"),
                usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            ),
        ]
        parser.tokenizers = {"test-model": mock_tokenizer}

        result = await parser.process_valid_record(create_test_record())

        # Each response has cumulative values - parser doesn't aggregate
        assert len(result.responses) == 3
        assert result.responses[0].usage.completion_tokens == 1
        assert result.responses[1].usage.completion_tokens == 2
        assert result.responses[2].usage.completion_tokens == 3

    @pytest.mark.asyncio
    async def test_passthrough_nested_reasoning(self, parser, mock_tokenizer):
        """Test parser passes through nested reasoning tokens."""
        parser.endpoint.extract_response_data.return_value = [
            ParsedResponse(
                perf_ns=100,
                data=TextResponseData(text="Answer"),
                usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 60,
                    "completion_tokens_details": {"reasoning_tokens": 50},
                },
            )
        ]
        parser.tokenizers = {"test-model": mock_tokenizer}

        result = await parser.process_valid_record(create_test_record())

        assert result.responses[0].usage.reasoning_tokens == 50

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "responses,expected_usage",
        [
            # No usage
            (
                [
                    ParsedResponse(
                        perf_ns=100, data=TextResponseData(text="Hi"), usage=None
                    )
                ],
                [None],
            ),
            # Partial usage
            (
                [
                    ParsedResponse(
                        perf_ns=100,
                        data=TextResponseData(text="A"),
                        usage={"prompt_tokens": 10},
                    )
                ],
                [{"prompt_tokens": 10}],
            ),
            # Mixed availability
            (
                [
                    ParsedResponse(
                        perf_ns=100,
                        data=TextResponseData(text="A"),
                        usage={"prompt_tokens": 10},
                    ),
                    ParsedResponse(
                        perf_ns=200, data=TextResponseData(text="B"), usage=None
                    ),
                    ParsedResponse(
                        perf_ns=300,
                        data=TextResponseData(text="C"),
                        usage={"completion_tokens": 5},
                    ),
                ],
                [{"prompt_tokens": 10}, None, {"completion_tokens": 5}],
            ),
        ],
    )
    async def test_passthrough_various_scenarios(
        self, parser, mock_tokenizer, responses, expected_usage
    ):
        """Test parser passes through various usage scenarios unchanged."""
        parser.endpoint.extract_response_data.return_value = responses
        parser.tokenizers = {"test-model": mock_tokenizer}

        result = await parser.process_valid_record(create_test_record())

        assert len(result.responses) == len(expected_usage)
        for i, expected in enumerate(expected_usage):
            actual_usage = result.responses[i].usage
            if expected is None:
                assert actual_usage is None
            else:
                assert actual_usage is not None
                assert dict(actual_usage) == expected
