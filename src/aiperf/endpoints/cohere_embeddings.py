# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    EmbeddingResponseData,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.common.types import RequestOutputT
from aiperf.endpoints.base_embeddings_endpoint import BaseEmbeddingsEndpoint


class CohereEmbeddingsEndpoint(BaseEmbeddingsEndpoint):
    """Cohere Embed v2 endpoint compatible with vLLM /v2/embed."""

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format payload for a Cohere Embed v2 request."""
        turn = self._validate_and_get_turn(request_info)
        texts = self._extract_texts(turn)
        images = self._extract_images(turn)

        if images:
            return self._build_payload(
                turn,
                embedding_types=["float"],
                inputs=self._build_multimodal_inputs(texts, images),
            )
        return self._build_payload(turn, embedding_types=["float"], texts=texts)

    def _build_multimodal_inputs(
        self, texts: list[str], images: list[str]
    ) -> list[dict[str, Any]]:
        """Build Cohere input objects for image-bearing requests.

        Mirrors NIMEmbeddingsEndpoint semantics for mixed turns: when texts and
        images are both present, they are treated as paired batch items and must
        therefore have equal lengths. Image-only requests still use Cohere
        `inputs`, because Cohere's dedicated `images` field is confusingly limited
        to a single image per request. See:
        https://docs.cohere.com/reference/embed
        """
        if texts and len(texts) != len(images):
            raise ValueError(
                "When both texts and images are provided, they must have the same length. "
                f"Got {len(texts)} texts and {len(images)} images."
            )

        if texts:
            return [
                {
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image}},
                    ]
                }
                for text, image in zip(texts, images, strict=False)
            ]

        return [
            {
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                ]
            }
            for image in images
        ]

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse Cohere Embed v2 responses."""
        json_obj = response.get_json()
        if not json_obj:
            self.debug(
                lambda: f"No JSON object found in response: {response.get_raw()}"
            )
            return None

        embeddings_obj = json_obj.get("embeddings")
        embeddings = self._extract_float_embeddings(embeddings_obj)
        usage = self._extract_usage(json_obj)

        if embeddings is None:
            self.debug(lambda: f"No float embeddings found in response: {json_obj}")
            if usage is None:
                return None
            return ParsedResponse(perf_ns=response.perf_ns, usage=usage)

        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=EmbeddingResponseData(embeddings=embeddings),
            usage=usage,
        )

    def _extract_float_embeddings(
        self, embeddings_obj: Any
    ) -> list[list[float]] | None:
        """Extract float embeddings from a Cohere response."""
        if not isinstance(embeddings_obj, dict):
            return None

        value = embeddings_obj.get("float")
        if not self._is_numeric_embedding_batch(value):
            return None
        return value

    def _extract_usage(self, json_obj: dict[str, Any]) -> dict[str, int] | None:
        """Normalize Cohere meta.billed_units into AIPerf's usage shape."""
        meta = json_obj.get("meta")
        if not isinstance(meta, dict):
            return None

        billed_units = meta.get("billed_units")
        if not isinstance(billed_units, dict):
            return None

        input_tokens = billed_units.get("input_tokens")
        if not isinstance(input_tokens, int):
            return None

        return {
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "total_tokens": input_tokens,
        }

    def _is_numeric_embedding_batch(self, value: Any) -> bool:
        """Return True if value is a non-empty list[list[number]]."""
        return (
            isinstance(value, list)
            and len(value) > 0
            and all(
                isinstance(vector, list)
                and all(isinstance(component, int | float) for component in vector)
                for vector in value
            )
        )
