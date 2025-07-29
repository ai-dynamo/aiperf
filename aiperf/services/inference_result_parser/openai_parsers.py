# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion import Completion
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.responses.response import Response as ResponsesModel
from pydantic import BaseModel

from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.enums import CaseInsensitiveStrEnum, EndpointType
from aiperf.common.factories import ResponseExtractorFactory
from aiperf.common.models import (
    InferenceServerResponse,
    RequestRecord,
    ResponseData,
    SSEMessage,
    TextResponse,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.utils import load_json_str

logger = logging.getLogger(__name__)


class OpenAIObject(CaseInsensitiveStrEnum):
    """Types of OpenAI objects."""

    CHAT_COMPLETION = "chat.completion"
    CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    LIST = "list"
    RESPONSE = "response"
    TEXT_COMPLETION = "text_completion"

    @classmethod
    def parse(cls, text: str) -> BaseModel:
        """Attempt to parse a string into an OpenAI object.

        Raises:
            ValueError: If the object is invalid.
        """
        try:
            obj = load_json_str(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid OpenAI object: {text}") from e

        # Mapping of OpenAI object types to their corresponding Pydantic models.
        _object_mapping: dict[str, type[BaseModel]] = {
            cls.CHAT_COMPLETION: ChatCompletion,
            cls.CHAT_COMPLETION_CHUNK: ChatCompletionChunk,
            cls.COMPLETION: Completion,
            cls.RESPONSE: ResponsesModel,
            cls.TEXT_COMPLETION: Completion,  # Alias for vLLM compatibility
        }

        obj_type = obj.get("object")
        if obj_type is None:
            raise ValueError(f"Invalid OpenAI object: {obj}")
        if obj_type == cls.LIST:
            return cls.parse_list(obj)
        if obj_type not in _object_mapping:
            raise ValueError(f"Invalid OpenAI object type: {obj_type}")
        try:
            # Hotfix: vLLM does not always include a finish_reason, which Pydantic requires.
            # Without this code, model_validate will raise an objection due to the missing finish_reason.
            if obj_type == cls.TEXT_COMPLETION:
                for choice in obj.get("choices", []):
                    if choice.get("finish_reason") is None:
                        choice["finish_reason"] = "stop"
            return _object_mapping[obj_type].model_validate(obj)
        except Exception as e:
            raise ValueError(f"Invalid OpenAI object: {text}") from e

    @classmethod
    def parse_list(cls, obj: Any) -> BaseModel:
        """Attempt to parse a string into an OpenAI object from a list.

        Raises:
            ValueError: If the object is invalid.
        """
        data = obj.get("data", [])
        if all(
            isinstance(item, dict) and item.get("object") == cls.EMBEDDING
            for item in data
        ):
            return CreateEmbeddingResponse.model_validate(obj)
        else:
            raise ValueError(f"Receive invalid list in response: {obj}")


@ResponseExtractorFactory.register_all(
    EndpointType.OPENAI_CHAT_COMPLETIONS,
    EndpointType.OPENAI_COMPLETIONS,
    EndpointType.OPENAI_EMBEDDINGS,
    EndpointType.OPENAI_RESPONSES,
)
class OpenAIResponseExtractor:
    """Extractor for OpenAI responses."""

    def __init__(self, model_endpoint: ModelEndpointInfo) -> None:
        """Create a new response extractor based on the provided configuration."""
        self.model_endpoint = model_endpoint

    def _parse_text_response(self, response: TextResponse) -> ResponseData | None:
        """Parse a TextResponse into a ResponseData object."""
        raw = response.text
        parsed = self._parse_text(raw)
        if parsed is None:
            return None

        return ResponseData(
            perf_ns=response.perf_ns,
            raw_text=[raw],
            parsed_text=[parsed],
            metadata={},
        )

    def _parse_sse_response(self, response: SSEMessage) -> ResponseData | None:
        """Parse a SSEMessage into a ResponseData object."""
        raw = response.extract_data_content()
        parsed = self._parse_sse(raw)
        if parsed is None or len(parsed) == 0:
            return None

        return ResponseData(
            perf_ns=response.perf_ns,
            raw_text=raw,
            parsed_text=parsed,
            metadata={},
        )

    def _parse_response(self, response: InferenceServerResponse) -> ResponseData | None:
        """Parse a response into a ResponseData object."""
        if isinstance(response, TextResponse):
            return self._parse_text_response(response)
        elif isinstance(response, SSEMessage):
            return self._parse_sse_response(response)

    async def extract_response_data(
        self, record: RequestRecord, tokenizer: Tokenizer | None
    ) -> list[ResponseData]:
        """Extract the text from a server response message."""
        results = []
        for response in record.responses:
            response_data = self._parse_response(response)
            if response_data is None:
                continue

            if tokenizer is not None:
                response_data.token_count = sum(
                    len(tokenizer.encode(text))
                    for text in response_data.parsed_text
                    if text is not None
                )
            results.append(response_data)
        return results

    def _parse_text(self, raw_text: str) -> Any | None:
        """Parse the text of the response."""
        if raw_text in ("", None, "[DONE]"):
            return None

        obj = OpenAIObject.parse(raw_text)

        # Dictionary mapping object types to their value extraction functions
        type_to_extractor = {
            # TODO: how to support multiple choices?
            ChatCompletion: lambda obj: obj.choices[0].message.content,
            # TODO: how to support multiple choices?
            ChatCompletionChunk: lambda obj: obj.choices[0].delta.content,
            # TODO: how to support multiple choices?
            Completion: lambda obj: obj.choices[0].text,
            CreateEmbeddingResponse: lambda obj: "",  # Don't store embedding data
            ResponsesModel: lambda obj: obj.output_text,
        }

        for obj_type, extractor in type_to_extractor.items():
            if isinstance(obj, obj_type):
                return extractor(obj)

        raise ValueError(f"Invalid OpenAI object: {raw_text}")

    def _parse_sse(self, raw_sse: list[str]) -> list[Any]:
        """Parse the SSE of the response."""
        result = []
        for sse in raw_sse:
            parsed = self._parse_text(sse)
            if parsed is None:
                continue
            result.append(parsed)
        return result
