# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.models import (
    InferenceServerResponse,
    ParsedResponse,
)
from aiperf.endpoints.openai_chat import ChatEndpoint


class ChatEmbeddingsEndpoint(ChatEndpoint):
    """Chat-style embeddings endpoint for vLLM multimodal embedding models.

    Required for vLLM as it is the only way to obtain multimodal embeddings.
    Uses chat messages format for requests but parses embeddings responses.
    """

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Embeddings response from chat-style request.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted embeddings
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        data = self.try_extract_embeddings(json_obj)
        if data:
            return ParsedResponse(perf_ns=response.perf_ns, data=data)
        return None
