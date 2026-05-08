# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import Media, RequestInfo, Turn
from aiperf.endpoints.base_endpoint import BaseEndpoint


class BaseEmbeddingsEndpoint(BaseEndpoint):
    """Shared utilities for embedding endpoints."""

    def _validate_and_get_turn(self, request_info: RequestInfo) -> Turn:
        """Validate request and return the single turn."""
        if len(request_info.turns) != 1:
            raise ValueError("Embeddings endpoint only supports one turn.")

        turn = request_info.turns[0]

        if turn.max_tokens:
            self.error("Max_tokens is provided but is not supported for embeddings.")

        return turn

    def _extract_contents(self, content_items: list[Media]) -> list[str]:
        """Flatten non-empty media contents while reusing BaseEndpoint helpers."""
        contents, _ = self.extract_named_contents(content_items)
        return [content for content in contents if content]

    def _extract_texts(self, turn: Turn) -> list[str]:
        """Extract non-empty text contents from a turn."""
        return self._extract_contents(turn.texts)

    def _extract_images(self, turn: Turn) -> list[str]:
        """Extract non-empty image contents from a turn."""
        return self._extract_contents(turn.images)

    def _build_payload(self, turn: Turn, **payload_fields: Any) -> dict[str, Any]:
        """Build an embeddings payload with the resolved model and endpoint extras."""
        payload: dict[str, Any] = {
            "model": turn.model or self.model_endpoint.primary_model_name,
            **payload_fields,
        }

        if self.model_endpoint.endpoint.extra:
            payload.update(self.model_endpoint.endpoint.extra)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload
