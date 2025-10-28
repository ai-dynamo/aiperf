# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import EndpointType
from aiperf.common.factories import EndpointFactory
from aiperf.common.models import (
    ParsedResponse,
)
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.models.record_models import RequestInfo
from aiperf.common.protocols import EndpointProtocol, InferenceServerResponse
from aiperf.endpoints.base_endpoint import BaseEndpoint


@implements_protocol(EndpointProtocol)
@EndpointFactory.register(EndpointType.HUGGINGFACE_GENERATE)
class HuggingfaceGenerate(BaseEndpoint):
    """Hugging Face Generate Endpoint."""

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        return EndpointMetadata(
            endpoint_path="/generate",
            supports_streaming=True,
            produces_tokens=True,
            tokenizes_input=True,
            metrics_title="LLM Metrics",
        )

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format Hugging Face Generate payload from RequestInfo.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            Hugging Face Generate payload
        """

        if not request_info.turns:
            raise ValueError("Hugging Face Generate requires at least one turn.")

        turn = request_info.turns[0]
        prompt = turn.texts[0].contents[0]

        payload = {"inputs": prompt, "parameters": {}}

        if turn.max_tokens:
            payload["parameters"]["max_new_tokens"] = turn.max_tokens

        if request_info.model_endpoint.endpoint.streaming:
            payload["parameters"]["stream"] = True

        if extra := request_info.model_endpoint.endpoint.extra:
            payload["parameters"].update(extra)

        self.debug(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse HF TGI response."""
        json_obj = response.get_json()
        if not json_obj:
            return None

        # TGI returns either a list or a dict
        if isinstance(json_obj, list):
            texts = [obj.get("generated_text", "") for obj in json_obj]
        else:
            texts = [json_obj.get("generated_text", "")]

        if not texts:
            return None

        data = self.make_text_response_data(texts[0])
        return ParsedResponse(perf_ns=response.perf_ns, data=data)
