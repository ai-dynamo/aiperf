# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    BaseResponseData,
    EmbeddingResponseData,
    InferenceServerResponse,
    Media,
    ModelEndpointInfo,
    ParsedResponse,
    RankingsResponseData,
    RequestInfo,
    RequestRecord,
    TextResponseData,
)
from aiperf.common.types import RequestOutputT


class BaseEndpoint(AIPerfLoggerMixin, ABC):
    """Base for all endpoints.

    Endpoints handle API-specific formatting and parsing.
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs):
        super().__init__(**kwargs)
        self.model_endpoint = model_endpoint

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Get endpoint headers (auth + user custom). Override to customize."""
        cfg = self.model_endpoint.endpoint
        headers = dict(cfg.headers) if cfg.headers else {}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        return headers

    def get_endpoint_params(self, request_info: RequestInfo) -> dict[str, str]:
        """Get endpoint URL query params (e.g., api-version). Override to customize."""
        cfg = self.model_endpoint.endpoint
        return dict(cfg.url_params) if cfg.url_params else {}

    @abstractmethod
    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format request payload from RequestInfo.

        Uses request_info.turns[0] as the turn data (currently hardcoded to first turn).
        """

    @abstractmethod
    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse response. Return None to skip."""

    def extract_response_data(self, record: RequestRecord) -> list[ParsedResponse]:
        """Extract parsed data from record.

        Args:
            record: Request record containing responses to parse

        Returns:
            List of successfully parsed responses
        """
        return [
            parsed
            for response in record.responses
            if (parsed := self.parse_response(response))
        ]

    @staticmethod
    def make_text_response_data(text: str | None) -> TextResponseData | None:
        """Make a TextResponseData object from a string or return None if the text is empty."""
        return TextResponseData(text=text) if text else None

    def auto_detect_and_extract(self, json_obj: dict) -> BaseResponseData | None:
        """Optional utility: Auto-detect response type and extract relevant data.

        Tries to extract data in this order: embeddings, rankings, text.
        Endpoints can use this as a fallback or for flexible response handling.

        Args:
            json_obj: JSON response object

        Returns:
            Typed response data object or None if not found
        """
        if data := self.try_extract_embeddings(json_obj):
            return data

        if data := self.try_extract_rankings(json_obj):
            return data

        if data := self.try_extract_text(json_obj):
            return data

        return None

    def try_extract_embeddings(self, json_obj: dict) -> EmbeddingResponseData | None:
        """Optional utility: Try to extract embeddings from common response formats.

        Supports:
        - OpenAI format: {"data": [{"embedding": [...], "object": "embedding"}, ...]}
        - Simple formats: {"embeddings": [[...], ...]} or {"embedding": [...]}

        Args:
            json_obj: JSON response object

        Returns:
            EmbeddingResponseData with extracted embeddings or None if not found
        """
        data = json_obj.get("data")
        if (
            isinstance(data, list)
            and data
            and isinstance(data[0], dict)
            and data[0].get("object") == "embedding"
        ):
            embeddings = [item["embedding"] for item in data if "embedding" in item]
            if embeddings:
                return EmbeddingResponseData(embeddings=embeddings)

        for field in ("embeddings", "embedding"):
            value = json_obj.get(field)
            if not (isinstance(value, list) and value):
                continue
            if isinstance(value[0], int | float):
                return EmbeddingResponseData(embeddings=[value])
            if isinstance(value[0], list):
                return EmbeddingResponseData(embeddings=value)

        return None

    def try_extract_rankings(self, json_obj: dict) -> RankingsResponseData | None:
        """Optional utility: Try to extract rankings from common response formats.

        Supports formats with "rankings" or "results" fields containing a list.

        Args:
            json_obj: JSON response object

        Returns:
            RankingsResponseData with extracted rankings or None if not found
        """
        for field in ("rankings", "results"):
            value = json_obj.get(field)
            if isinstance(value, list):
                return RankingsResponseData(rankings=value)
        return None

    def try_extract_text(self, json_obj: dict) -> TextResponseData | None:
        """Optional utility: Try to extract text from common response formats.

        Supports:
        - Simple fields: text, content, response, output, result
        - List of strings (joined without separator): {"text": ["A", "B", "C"]} -> "ABC"
        - OpenAI completions: {"choices": [{"text": "..."}]}
        - OpenAI chat (non-streaming): {"choices": [{"message": {"content": "..."}}]}
        - OpenAI chat (streaming): {"choices": [{"delta": {"content": "..."}}]}

        Args:
            json_obj: JSON response object

        Returns:
            TextResponseData with extracted text or None if not found
        """
        for field in ("text", "content", "response", "output", "result"):
            value = json_obj.get(field)
            if isinstance(value, str):
                return self.make_text_response_data(value)
            if (
                isinstance(value, list)
                and value
                and all(isinstance(item, str) for item in value)
            ):
                return self.make_text_response_data("".join(value))

        choices = json_obj.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if text := choice.get("text"):
                return self.make_text_response_data(text)
            # Non-streaming chat format
            message = choice.get("message")
            if message and (content := message.get("content")):
                return self.make_text_response_data(content)
            # Streaming chat format (delta)
            delta = choice.get("delta")
            if delta and (content := delta.get("content")):
                return self.make_text_response_data(content)

        return None

    def convert_to_response_data(self, value: Any) -> BaseResponseData | None:
        """Optional utility: Convert extracted value to appropriate response data type.

        Automatically determines the type based on the value structure:
        - list[list[float]] or list[float] -> EmbeddingResponseData
        - list[dict] -> RankingsResponseData
        - str -> TextResponseData

        Args:
            value: Extracted value from response

        Returns:
            Typed response data or None if conversion not possible
        """
        if value is None:
            return None

        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, list) and first and isinstance(first[0], int | float):
                return EmbeddingResponseData(embeddings=value)
            if isinstance(first, int | float):
                return EmbeddingResponseData(embeddings=[value])
            if isinstance(first, dict):
                return RankingsResponseData(rankings=value)

        if isinstance(value, str):
            return self.make_text_response_data(value)

        return None

    def extract_named_contents(
        self,
        content_items: list[Media],
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Extract contents and organize by name.

        Args:
            content_items: List of content items (texts, images, audios, videos)

        Returns:
            Tuple of (all_contents, contents_by_name)
        """
        all_contents = []
        by_name: dict[str, list[str]] = {}

        for item in content_items:
            if not item.contents:
                continue
            all_contents.extend(item.contents)
            if item.name:
                by_name.setdefault(item.name, []).extend(item.contents)

        return all_contents, by_name
