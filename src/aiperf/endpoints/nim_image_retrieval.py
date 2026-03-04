# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from aiperf.common.models import ParsedResponse
from aiperf.common.models.record_models import (
    ImageRetrievalResponseData,
    InferenceServerResponse,
    RequestInfo,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint


class ImageRetrievalEndpoint(BaseEndpoint):
    """NIM Image Retrieval endpoint."""

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format payload for an image retrieval request."""
        if len(request_info.turns) != 1:
            raise ValueError("Image Retrieval endpoint only supports one turn.")

        turn = request_info.turns[0]
        model_endpoint = request_info.model_endpoint

        if turn.max_tokens:
            self.warning(
                "Max_tokens is provided but is not supported for Image Retrieval."
            )

        if not turn.images:
            raise ValueError("Image Retrieval request requires at least one image.")

        payload: dict[str, Any] = {
            "input": [
                {"type": "image_url", "url": content}
                for img in turn.images
                if img.contents
                for content in img.contents
                if content
            ],
        }

        if not payload["input"]:
            raise ValueError(
                "No valid image content found. All images have empty contents or "
                "empty content values."
            )

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.trace(lambda: f"Formatted Image Retrieval payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse NIM Image Retrieval response."""
        json_obj = response.get_json()
        if not json_obj:
            self.debug(
                lambda: f"No JSON object found in response: {response.get_raw()}"
            )
            return None

        data = json_obj.get("data", None)
        if not data:
            self.debug(lambda: f"No data found in response: {json_obj}")
            return None

        return ParsedResponse(
            perf_ns=response.perf_ns, data=ImageRetrievalResponseData(data=data)
        )
