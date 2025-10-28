# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Any

from aiperf.common.enums import EndpointType
from aiperf.common.factories import EndpointFactory
from aiperf.common.models.model_endpoint_info import EndpointMetadata
from aiperf.common.models.record_models import TextResponseData
from aiperf.common.protocols import InferenceServerResponse, ParsedResponse
from aiperf.endpoints.base_endpoint import BaseEndpoint

LOGGER = logging.getLogger(__name__)


@EndpointFactory.register(EndpointType.HUGGINGFACE_GENERATE)
class HuggingFaceGenerateEndpoint(BaseEndpoint):
    """Hugging Face TGI endpoint supporting both /generate and /generate_stream."""

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        return EndpointMetadata(
            endpoint_path="/generate",
            supports_streaming=True,
            produces_tokens=True,
            tokenizes_input=True,
            metrics_title="LLM Metrics",
        )

    def _get_request_path(self, stream: bool = False) -> str:
        """Return /generate or /generate_stream depending on streaming mode."""
        base_path = self.metadata().endpoint_path
        return f"{base_path}_stream" if stream else base_path

    def format_payload(self, request_info) -> dict[str, Any]:
        """Format request payload for Hugging Face TGI."""
        turn = request_info.turns[-1]
        prompt = turn.texts[0].contents[0]

        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {},
        }

        if getattr(turn, "max_tokens", None):
            payload["parameters"]["max_new_tokens"] = turn.max_tokens

        if self.model_endpoint.endpoint.extra:
            payload["parameters"].update(self.model_endpoint.endpoint.extra)

        payload["stream"] = bool(
            getattr(self.model_endpoint.endpoint, "streaming", False)
        )
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse JSON from TGI response into ParsedResponse."""
        json_obj = response.get_json()
        if not json_obj:
            return None

        if isinstance(json_obj, list):
            full_text = "".join(
                obj["generated_text"]
                for obj in json_obj
                if isinstance(obj, dict) and obj.get("generated_text")
            )
        elif isinstance(json_obj, dict):
            full_text = json_obj.get("generated_text") or ""
        else:
            return None

        if not full_text.strip():
            return None

        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=TextResponseData(text=full_text),
        )

    def send_request(self, request_info) -> InferenceServerResponse:
        """Send POST request to TGI /generate or /generate_stream."""
        payload = self.format_payload(request_info)
        stream = payload.get("stream", False)
        path = self._get_request_path(stream)
        base_url = self.model_endpoint.endpoint.base_url.rstrip("/")
        url = f"{base_url}{path}"

        LOGGER.debug(
            f"Sending {'streaming' if stream else 'non-streaming'} request to {url}"
        )
        return self.transport.post(url, json=payload)

    def stream_response(self, response_iter):
        """Iterate token-by-token from /generate_stream SSE output."""
        for event in response_iter:
            try:
                if not event.data:
                    continue
                json_obj = event.json()
                token_text = json_obj.get("token", {}).get("text", "")
                if token_text:
                    yield TextResponseData(text=token_text)
            except Exception as e:
                LOGGER.warning(f"Error parsing stream event: {e}")
                continue
