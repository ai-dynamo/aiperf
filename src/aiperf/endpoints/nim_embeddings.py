# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import RequestInfo
from aiperf.common.types import RequestOutputT
from aiperf.endpoints.openai_embeddings import EmbeddingsEndpoint


class NIMEmbeddingsEndpoint(EmbeddingsEndpoint):
    """NVIDIA NIM Embeddings endpoint.

    Extends the OpenAI Embeddings endpoint with multimodal support for images.
    NIM Embeddings API is a superset of the OpenAI Embeddings API.
    """

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format payload for a NIM embeddings request with multimodal support.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            NIM Embeddings API payload (supports text and/or images)
        """
        turn = self._validate_and_get_turn(request_info)

        # Extract text contents
        texts = [content for text in turn.texts for content in text.contents if content]

        # Extract images (list of data URL strings)
        images = [
            image_content
            for image in turn.images
            for image_content in image.contents
            if image_content
        ]

        # Determine inputs based on content type
        if texts and images:
            if len(texts) != len(images):
                raise ValueError(
                    f"When both texts and images are provided, they must have the same length. "
                    f"Got {len(texts)} texts and {len(images)} images."
                )
            inputs: list[Any] = [
                f"{text} {image}" for text, image in zip(texts, images, strict=False)
            ]
        elif images:
            inputs = images
        else:
            inputs = texts

        return self._build_payload(turn, request_info.model_endpoint, inputs)
