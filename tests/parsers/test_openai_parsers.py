# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest

from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models import (
    ParsedResponse,
    RequestRecord,
    SSEMessage,
    TextResponse,
    TextResponseData,
)
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

    def create_sse_message(self, chunk: str, perf_ns=2000000) -> MagicMock:
        """Create a mock SSEMessage with specified chunk contents."""
        sse_message = MagicMock(spec=SSEMessage)
        sse_message.extract_data_content.return_value = self.chat_completion_chunk_json(
            chunk
        )
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
        result = extractor._parse_raw_text(text)
        assert result is None

    @pytest.mark.parametrize("content", ["", None])
    def test_parse_text_with_empty_content_returns_none(self, extractor, content):
        """Test that valid chat completion with empty/null content returns None."""
        chat_completion_json = self.chat_completion_json(content)

        result = extractor._parse_raw_text(chat_completion_json)
        assert result is None

    @pytest.mark.parametrize("content", ["", None])
    def test_parse_text_with_empty_chunk_content_returns_none(self, extractor, content):
        """Test that valid chat completion chunk with empty/null delta content returns None."""
        chunk_json = self.chat_completion_chunk_json(content)

        result = extractor._parse_raw_text(chunk_json)
        assert result is None

    def test_parse_text_with_valid_content_returns_content(self, extractor):
        """Test that valid chat completion with actual content returns the content."""
        test_content = "Hello, how can I help you?"
        chat_completion_json = self.chat_completion_json(test_content)

        result = extractor._parse_raw_text(chat_completion_json)
        assert result is not None
        assert isinstance(result, TextResponseData)
        assert result.get_text() == test_content

    def test_parse_text_with_valid_chunk_content_returns_content(self, extractor):
        """Test that valid chat completion chunk with actual delta content returns the content."""
        test_content = "Stream chunk content"
        chunk_json = self.chat_completion_chunk_json(test_content)

        result = extractor._parse_raw_text(chunk_json)
        assert result is not None
        assert isinstance(result, TextResponseData)
        assert result.get_text() == test_content

    def test_parse_text_response_with_empty_content_returns_none(self, extractor):
        """Test that TextResponse with empty content is ignored."""
        text_response = self.create_raw_text_response("")

        result = extractor._parse_response(text_response)
        assert result is None

    def test_parse_text_response_with_valid_content_returns_response_data(
        self, extractor
    ):
        """Test that TextResponse with valid content returns ResponseData."""
        test_content = "Valid response"
        text_response = self.create_text_response(test_content, perf_ns=1)

        result = extractor._parse_response(text_response)

        assert result is not None
        assert isinstance(result, ParsedResponse)
        assert result.data.get_text() == test_content
        assert result.perf_ns == 1

    def test_parse_sse_response_with_empty_chunks_returns_none(self, extractor):
        """Test that SSEMessage with empty chunks is ignored."""
        sse_message = self.create_sse_message("")

        result = extractor._parse_response(sse_message)
        assert result is None

    def test_parse_sse_response_with_mixed_chunks_filters_empty(self, extractor):
        """Test that SSEMessage filters out empty chunks but keeps valid ones."""
        sse_message = self.create_sse_message("Valid chunk", perf_ns=1)

        result = extractor._parse_response(sse_message)

        assert result is not None
        assert isinstance(result, ParsedResponse)
        assert result.data.get_text() == "Valid chunk"
        assert result.perf_ns == 1

    @pytest.mark.asyncio
    async def test_extract_response_data_filters_empty_responses(self, extractor):
        """Test that extract_response_data filters out responses with empty content."""
        request = self.create_request_record(
            self.create_raw_text_response("", perf_ns=1000000),  # Raw empty text
            self.create_text_response("Valid response", perf_ns=2000000),
        )

        results = await extractor.extract_response_data(request)

        # Should only return the valid response, empty one should be filtered out
        assert len(results) == 1
        assert results[0].data.get_text() == "Valid response"
        assert results[0].perf_ns == 2000000

    @pytest.mark.asyncio
    async def test_extract_response_data_handles_mixed_response_types(self, extractor):
        """Test that extract_response_data handles mixed TextResponse and SSEMessage types."""
        request = self.create_request_record(
            self.create_text_response("Text response", perf_ns=1000000),
            self.create_sse_message("SSE chunk", perf_ns=2000000),
        )

        results = await extractor.extract_response_data(request)

        # Should return both responses
        assert len(results) == 2
        assert results[0].data.get_text() == "Text response"
        assert results[0].perf_ns == 1000000
        assert results[1].data.get_text() == "SSE chunk"
        assert results[1].perf_ns == 2000000

    @pytest.mark.asyncio
    async def test_extract_response_data_with_complex_sse_filtering(self, extractor):
        """Test extract_response_data with complex SSE message filtering."""
        request = self.create_request_record(
            self.create_text_response("Valid text response", perf_ns=1),
            self.create_sse_message("", perf_ns=2),
            self.create_sse_message("Valid chunk 1", perf_ns=3),
            self.create_sse_message("", perf_ns=4),
            self.create_sse_message("Valid chunk 2", perf_ns=5),
            self.create_sse_message("", perf_ns=6),
            self.create_raw_text_response("", perf_ns=7),
        )

        results = await extractor.extract_response_data(request)

        # Should return text response + filtered SSE response (empty raw_text filtered out)
        assert len(results) == 3
        assert results[0].data.get_text() == "Valid text response"
        assert results[0].perf_ns == 1
        assert results[1].data.get_text() == "Valid chunk 1"
        assert results[1].perf_ns == 3
        assert results[2].data.get_text() == "Valid chunk 2"
        assert results[2].perf_ns == 5


class TestRankingsParser(TestOpenAIResponseExtractor):
    """Test cases for RankingsParser."""

    @pytest.fixture
    def extractor(self):
        """Create an OpenAIResponseExtractor instance."""
        mock_endpoint = MagicMock(spec=ModelEndpointInfo)
        return OpenAIResponseExtractor(mock_endpoint)

    def rankings_response_json(self, rankings_data) -> str:
        """Generate rankings response JSON with specified rankings data."""
        response = {
            "object": "rankings",
            "data": rankings_data,
            "model": "test-rankings-model",
            "usage": {"total_tokens": 50},
        }
        return json.dumps(response)

    @pytest.mark.asyncio
    async def test_ranking_response_parsing(self, extractor):
        rankings_data = [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.87},
            {"index": 2, "relevance_score": 0.72},
        ]

        text_response = self.create_raw_text_response(
            self.rankings_response_json(rankings_data)
        )
        request = self.create_request_record(text_response)

        results = await extractor.extract_response_data(request)

        assert len(results) == 1
        result = results[0]

        assert hasattr(result.data, "rankings")
        assert result.data.rankings == rankings_data
        assert len(result.data.rankings) == 3
        assert result.data.rankings[0]["relevance_score"] == 0.95

    @pytest.mark.asyncio
    async def test_ranking_response_empty_data(self, extractor):
        text_response = self.create_raw_text_response(self.rankings_response_json([]))
        request = self.create_request_record(text_response)

        results = await extractor.extract_response_data(request)

        # Empty rankings should be filtered out, resulting in no results
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_rankings_response_with_metadata(self, extractor):
        rankings_data = [
            {
                "index": 0,
                "relevance_score": 0.95,
                "document_id": "doc1",
                "snippet": "This is a relevant passage",
            },
            {
                "index": 1,
                "relevance_score": 0.87,
                "document_id": "doc2",
                "snippet": "Another relevant passage",
            },
        ]

        text_response = self.create_raw_text_response(
            self.rankings_response_json(rankings_data)
        )
        request = self.create_request_record(text_response)

        results = await extractor.extract_response_data(request)

        assert len(results) == 1
        result = results[0]

        assert result.data.rankings == rankings_data
        assert result.data.rankings[0]["document_id"] == "doc1"
        assert result.data.rankings[1]["snippet"] == "Another relevant passage"


class TestListParserWithEmbeddings(TestOpenAIResponseExtractor):
    """Test cases for ListParser with embedding responses."""

    @pytest.fixture
    def extractor(self):
        """Create an OpenAIResponseExtractor instance."""
        mock_endpoint = MagicMock(spec=ModelEndpointInfo)
        return OpenAIResponseExtractor(mock_endpoint)

    def embedding_list_response_json(self, embeddings_data) -> str:
        """Generate embedding list response JSON."""
        response = {
            "object": "list",
            "data": embeddings_data,
            "model": "test-embedding-model",
            "usage": {"total_tokens": 25},
        }
        return json.dumps(response)

    @pytest.mark.asyncio
    async def test_embedding_list_parsing(self, extractor):
        """Test parsing of embedding list response."""
        embeddings_data = [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]},
            {"object": "embedding", "index": 1, "embedding": [0.5, 0.6, 0.7, 0.8]},
        ]

        text_response = self.create_raw_text_response(
            self.embedding_list_response_json(embeddings_data)
        )
        request = self.create_request_record(text_response)

        results = await extractor.extract_response_data(request)

        assert len(results) == 1
        result = results[0]

        # Should be EmbeddingResponseData
        assert hasattr(result.data, "embeddings")
        assert len(result.data.embeddings) == 2
        assert result.data.embeddings[0] == [0.1, 0.2, 0.3, 0.4]
        assert result.data.embeddings[1] == [0.5, 0.6, 0.7, 0.8]

    @pytest.mark.asyncio
    async def test_invalid_list_response(self, extractor):
        """Test that invalid list response raises ValueError."""
        invalid_data = [{"object": "invalid", "data": "something"}]

        text_response = self.create_raw_text_response(
            self.embedding_list_response_json(invalid_data)
        )
        request = self.create_request_record(text_response)

        with pytest.raises(ValueError, match="Received invalid list in response"):
            await extractor.extract_response_data(request)
