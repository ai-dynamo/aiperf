# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest

from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models import RequestRecord, ResponseData, SSEMessage, TextResponse
from aiperf.parsers.openai_parsers import OpenAIResponseExtractor


class TestOpenAIResponseExtractor:
    """Test cases for OpenAIResponseExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create an OpenAIResponseExtractor instance."""
        mock_endpoint = MagicMock(spec=ModelEndpointInfo)
        return OpenAIResponseExtractor(mock_endpoint)

    def chat_completion_json(self, content) -> str:
        """Generate chat completion JSON with specified content and finish reason."""
        completion = {
            "id": "test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        assert completion["choices"][0]["message"]["content"] == content
        return json.dumps(completion)

    def chat_completion_chunk_json(self, content, stop=True) -> str:
        """Generate chat completion chunk JSON with specified delta content and finish reason."""
        chunk = {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": "stop" if stop else None,
                }
            ],
        }
        assert chunk["choices"][0]["delta"]["content"] == content
        return json.dumps(chunk)

    def create_raw_text_response(self, content, perf_ns=1000000) -> MagicMock:
        """Create a mock TextResponse with specified content."""
        text_response = MagicMock(spec=TextResponse)
        text_response.text = content
        text_response.perf_ns = perf_ns
        return text_response

    def create_text_response(self, content, perf_ns=1000000) -> MagicMock:
        """Create a mock TextResponse with specified content."""
        text_response = MagicMock(spec=TextResponse)
        text_response.text = self.chat_completion_json(content)
        text_response.perf_ns = perf_ns
        return text_response

    def create_sse_message(self, chunks, perf_ns=2000000) -> MagicMock:
        """Create a mock SSEMessage with specified chunk contents."""
        sse_message = MagicMock(spec=SSEMessage)
        if isinstance(chunks, str):
            # Single chunk
            sse_message.extract_data_content.return_value = [
                self.chat_completion_chunk_json(chunks)
            ]
        else:
            # Multiple chunks
            sse_message.extract_data_content.return_value = [
                self.chat_completion_chunk_json(chunk) for chunk in chunks
            ]
        sse_message.perf_ns = perf_ns
        return sse_message

    def create_request_record(self, *responses) -> MagicMock:
        """Create a mock RequestRecord with specified responses."""
        record = MagicMock(spec=RequestRecord)
        record.responses = list(responses)
        return record

    @pytest.mark.parametrize("text", ["[DONE]", "", None])
    def test_parse_text_returns_none(self, extractor, text):
        """Test that _parse_text returns None for '[DONE]' marker, empty string, and None."""
        result = extractor._parse_text(text)
        assert result is None

    @pytest.mark.parametrize("content", ["", None])
    def test_parse_text_with_empty_content_returns_none(self, extractor, content):
        """Test that valid chat completion with empty/null content returns None."""
        chat_completion_json = self.chat_completion_json(content)

        result = extractor._parse_text(chat_completion_json)
        assert result is None

    @pytest.mark.parametrize("content", ["", None])
    def test_parse_text_with_empty_chunk_content_returns_none(self, extractor, content):
        """Test that valid chat completion chunk with empty/null delta content returns None."""
        chunk_json = self.chat_completion_chunk_json(content)

        result = extractor._parse_text(chunk_json)
        assert result is None

    def test_parse_text_with_valid_content_returns_content(self, extractor):
        """Test that valid chat completion with actual content returns the content."""
        test_content = "Hello, how can I help you?"
        chat_completion_json = self.chat_completion_json(test_content)

        result = extractor._parse_text(chat_completion_json)
        assert result == test_content

    def test_parse_text_with_valid_chunk_content_returns_content(self, extractor):
        """Test that valid chat completion chunk with actual delta content returns the content."""
        test_content = "Stream chunk content"
        chunk_json = self.chat_completion_chunk_json(test_content)

        result = extractor._parse_text(chunk_json)
        assert result == test_content

    def test_parse_text_response_with_empty_content_returns_none(self, extractor):
        """Test that TextResponse with empty content is ignored."""
        text_response = self.create_raw_text_response("")

        result = extractor._parse_text_response(text_response)
        assert result is None

    def test_parse_text_response_with_valid_content_returns_response_data(
        self, extractor
    ):
        """Test that TextResponse with valid content returns ResponseData."""
        test_content = "Valid response"
        text_response = self.create_text_response(test_content)

        result = extractor._parse_text_response(text_response)

        assert result is not None
        assert isinstance(result, ResponseData)
        assert result.parsed_text == [test_content]
        assert result.perf_ns == 1000000

    def test_parse_sse_response_with_empty_chunks_returns_none(self, extractor):
        """Test that SSEMessage with empty chunks is ignored."""
        sse_message = self.create_sse_message("")

        result = extractor._parse_sse_response(sse_message)
        assert result is None

    def test_parse_sse_response_with_mixed_chunks_filters_empty(self, extractor):
        """Test that SSEMessage filters out empty chunks but keeps valid ones."""
        sse_message = self.create_sse_message(["", "Valid chunk"])

        result = extractor._parse_sse_response(sse_message)

        assert result is not None
        assert isinstance(result, ResponseData)
        assert result.parsed_text == ["Valid chunk"]
        assert result.perf_ns == 2000000

    @pytest.mark.asyncio
    async def test_extract_response_data_filters_empty_responses(self, extractor):
        """Test that extract_response_data filters out responses with empty content."""
        request = self.create_request_record(
            self.create_raw_text_response("", perf_ns=1000000),  # Raw empty text
            self.create_text_response("Valid response", perf_ns=2000000),
        )

        results = await extractor.extract_response_data(request, None)

        # Should only return the valid response, empty one should be filtered out
        assert len(results) == 1
        assert results[0].parsed_text == ["Valid response"]
        assert results[0].perf_ns == 2000000

    @pytest.mark.asyncio
    async def test_extract_response_data_handles_mixed_response_types(self, extractor):
        """Test that extract_response_data handles mixed TextResponse and SSEMessage types."""
        request = self.create_request_record(
            self.create_text_response("Text response", perf_ns=1000000),
            self.create_sse_message("SSE chunk", perf_ns=2000000),
        )

        results = await extractor.extract_response_data(request, None)

        # Should return both responses
        assert len(results) == 2
        assert results[0].parsed_text == ["Text response"]
        assert results[0].perf_ns == 1000000
        assert results[1].parsed_text == ["SSE chunk"]
        assert results[1].perf_ns == 2000000

    @pytest.mark.asyncio
    async def test_extract_response_data_with_complex_sse_filtering(self, extractor):
        """Test extract_response_data with complex SSE message filtering."""
        request = self.create_request_record(
            self.create_text_response("Valid text response", perf_ns=1000000),
            self.create_sse_message(
                ["", "Valid chunk 1", "", "Valid chunk 2"], perf_ns=3000000
            ),
            self.create_raw_text_response("", perf_ns=4000000),  # Should be filtered
        )

        results = await extractor.extract_response_data(request, None)

        # Should return text response + filtered SSE response (empty raw_text filtered out)
        assert len(results) == 2
        assert results[0].parsed_text == ["Valid text response"]
        assert results[0].perf_ns == 1000000
        assert results[1].parsed_text == [
            "Valid chunk 1",
            "Valid chunk 2",
        ]  # Empty chunks filtered
        assert results[1].perf_ns == 3000000
