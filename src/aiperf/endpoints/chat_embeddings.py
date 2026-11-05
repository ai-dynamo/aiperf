# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.models import ParsedResponse
from aiperf.common.types import JsonObject
from aiperf.endpoints.openai_chat import ChatEndpoint


class ChatEmbeddingsEndpoint(ChatEndpoint):
    """Chat-style embeddings endpoint for vLLM multimodal embedding models.

    Required for vLLM as it is the only way to obtain multimodal embeddings.
    Uses chat messages format for requests but parses embeddings responses.
    """

    def _parse_json_response(
        self, perf_ns: int, json_obj: JsonObject
    ) -> ParsedResponse | None:
        """Parse embeddings through the shared single-response and worker paths."""
        data = self.try_extract_embeddings(json_obj)
        if data:
            usage = json_obj.get("usage")
            return ParsedResponse(
                perf_ns=perf_ns,
                data=data,
                usage=(usage or None) if isinstance(usage, dict) else None,
            )
        return None
