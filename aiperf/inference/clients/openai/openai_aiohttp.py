# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
from abc import ABC
from typing import Any

from aiperf.common.enums import EndpointType
from aiperf.common.factories import InferenceClientFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ErrorDetails, RequestRecord
from aiperf.inference.clients.http.aiohttp_client import AioHttpClientMixin
from aiperf.inference.model_endpoint_info import ModelEndpointInfo


@InferenceClientFactory.register_all(
    EndpointType.OPENAI_CHAT_COMPLETIONS,
    EndpointType.OPENAI_COMPLETIONS,
    EndpointType.OPENAI_EMBEDDINGS,
    EndpointType.OPENAI_RESPONSES,
)
class OpenAIClientAioHttp(AioHttpClientMixin, AIPerfLoggerMixin, ABC):
    """Inference client for OpenAI based requests using aiohttp."""

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs) -> None:
        super().__init__(model_endpoint, **kwargs)
        self.model_endpoint = model_endpoint

    def get_headers(self, model_endpoint: ModelEndpointInfo) -> dict[str, str]:
        """Get the headers for the given endpoint."""

        accept = (
            "text/event-stream"
            if model_endpoint.endpoint.streaming
            else "application/json"
        )

        headers = {
            "User-Agent": "aiperf/1.0",
            "Content-Type": "application/json",
            "Accept": accept,
        }
        if model_endpoint.endpoint.api_key:
            headers["Authorization"] = f"Bearer {model_endpoint.endpoint.api_key}"
        if model_endpoint.endpoint.headers:
            headers.update(model_endpoint.endpoint.headers)
        return headers

    def get_url(self, model_endpoint: ModelEndpointInfo) -> str:
        """Get the URL for the given endpoint."""
        url = model_endpoint.url
        if not url.startswith("http"):
            url = f"http://{url}"
        return url

    async def send_request(
        self,
        model_endpoint: ModelEndpointInfo,
        payload: dict[str, Any],
    ) -> RequestRecord:
        """Send OpenAI request using aiohttp."""

        # capture start time before request is sent in the case of an error
        start_perf_ns = time.perf_counter_ns()
        try:
            self.debug(
                lambda: f"Sending OpenAI request to {model_endpoint.url}, payload: {payload}"
            )

            record = await self.post_request(
                self.get_url(model_endpoint),
                json.dumps(payload),
                self.get_headers(model_endpoint),
            )
            record.request = payload

        except Exception as e:
            record = RequestRecord(
                request=payload,
                start_perf_ns=start_perf_ns,
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails(type=e.__class__.__name__, message=str(e)),
            )
            self.exception(f"Error in OpenAI request: {e.__class__.__name__} {str(e)}")

        return record
