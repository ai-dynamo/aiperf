# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from aiperf.common.record_models import (
    RequestRecord,
    ResponseData,
    SSEMessage,
    TextResponse,
)
from aiperf.common.utils import load_json_str

logger = logging.getLogger(__name__)


class OpenAIResponseExtractor:
    """Extractor for OpenAI responses."""

    async def extract_response_data(self, record: RequestRecord) -> list[ResponseData]:
        """Extract the text from a server response message."""
        results = []
        for response in record.responses:
            if isinstance(response, TextResponse):
                raw = response.text
                parsed, metadata = self._parse_text(raw)
                if parsed is not None:
                    results.append(
                        ResponseData(
                            perf_ns=response.perf_ns,
                            raw_text=[raw],
                            parsed_text=parsed,
                            metadata=metadata or {},
                        )
                    )
            elif isinstance(response, SSEMessage):
                raw = response.extract_data_content()
                parsed, metadata = self._parse_sse(raw)
                if parsed is not None:
                    results.append(
                        ResponseData(
                            perf_ns=response.perf_ns,
                            raw_text=raw,
                            parsed_text=parsed,
                            metadata=metadata or {},
                        )
                    )
            else:
                raise ValueError(f"Unsupported response type: {type(response)}")
        return results

    def _parse_text(
        self, raw_text: str
    ) -> tuple[list[str] | None, dict[str, Any] | None]:
        """Parse the text of the response."""
        if raw_text in ("", None, "[DONE]"):
            return None, {}

        js = load_json_str(raw_text)
        if "choices" not in js:
            raise ValueError(f"Invalid OpenAI response: {js}")

        # TODO: how to support multiple choices?

        metadata = {
            "id": js["id"],
            "model": js["model"],
            "object": js["object"],
        }
        if "usage" in js:
            metadata["usage"] = js["usage"]

        # TODO: Parse based on the object type

        choice = js["choices"][0]
        if "text" in choice:
            return [choice["text"]], metadata
        elif "delta" in choice:
            if "content" in choice["delta"] and choice["delta"]["content"] not in (
                None,
                "",
            ):
                # logger.debug("Parsing delta: %s", choice["delta"])
                return [choice["delta"]["content"]], metadata

        elif "message" in choice:
            if choice["message"]["role"] == "assistant" and choice["message"][
                "content"
            ] not in (None, ""):
                metadata["role"] = "assistant"
                return [choice["message"]["content"]], metadata

        else:
            raise ValueError(f"Invalid OpenAI response: {js}")
        return None, metadata

    def _parse_sse(
        self, raw_sse: list[str]
    ) -> tuple[list[str] | None, dict[str, Any] | None]:
        """Parse the SSE of the response."""
        result = []
        all_metadata = {}
        for sse in raw_sse:
            parsed, metadata = self._parse_text(sse)
            if parsed is None:
                continue
            result.extend(parsed)

            # TODO: how to handle multiple metadata?
            if metadata is None:
                continue

            for k, v in metadata.items():
                if k not in all_metadata:
                    all_metadata[k] = v
                else:
                    if not isinstance(all_metadata[k], list):
                        if all_metadata[k] == v:
                            continue
                        all_metadata[k] = [all_metadata[k], v]
                    else:
                        if v not in all_metadata[k]:
                            all_metadata[k].append(v)
        # logger.debug(f"All metadata: {all_metadata}")
        return result, all_metadata
