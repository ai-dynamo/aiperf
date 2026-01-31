# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process fake transport for testing without network overhead.

The fake bypasses HTTP entirely, directly invoking aiperf_mock_server logic for
fast, isolated testing with configurable latency simulation.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import orjson
from aiperf_mock_server.app import (
    _build_chat_response_data,
    _build_cohere_ranking_response_data,
    _build_completion_response_data,
    _build_embedding_response_data,
    _build_hf_tei_ranking_response_data,
    _build_image_response_data,
    _build_nim_ranking_response_data,
    _build_solido_rag_response_data,
    _build_tgi_response_data,
    _compute_ranked_scores,
    _wait_for_processing,
)
from aiperf_mock_server.config import MockServerConfig
from aiperf_mock_server.models import (
    ChatCompletionRequest,
    CohereRerankRequest,
    CompletionRequest,
    EmbeddingRequest,
    HFTEIRerankRequest,
    ImageGenerationRequest,
    RankingRequest,
    SolidoRAGRequest,
    TGIGenerateRequest,
)
from aiperf_mock_server.utils import (
    RequestCtx,
    make_ctx,
    stream_chat_completion,
    stream_text_completion,
    stream_tgi_completion,
)
from pydantic import BaseModel

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import (
    ErrorDetails,
    RequestInfo,
    RequestRecord,
    SSEMessage,
    TextResponse,
)
from aiperf.common.types import RequestInputT
from aiperf.common.utils import yield_to_event_loop
from aiperf.plugin import plugins
from aiperf.plugin.enums import EndpointType, PluginType, TransportType
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import BaseTransport, FirstTokenCallback

if TYPE_CHECKING:
    from aiperf.common.models.model_endpoint_info import ModelEndpointInfo

BuildResponseFn: TypeAlias = Callable[..., dict[str, Any]]
StreamFn: TypeAlias = Callable[[RequestCtx, str, bool], AsyncGenerator[bytes, None]]


@dataclass(frozen=True, slots=True)
class HandlerInput:
    """Common inputs for all request handlers."""

    start_perf_ns: int
    ctx: RequestCtx
    endpoint_path: str
    req: BaseModel
    stream_fn: StreamFn | None = None
    build_response: BuildResponseFn | None = None
    first_token_callback: FirstTokenCallback | None = None


HandlerFn: TypeAlias = Callable[[HandlerInput], Awaitable[RequestRecord]]


class FakeTransport(BaseTransport):
    """In-process fake transport that bypasses HTTP (test double: Fake).

    Directly invokes aiperf_mock_server logic for fast, isolated testing.
    Supports all endpoint types: chat, completions, embeddings, rankings, images.
    """

    # Default config with zero latency for fast testing
    _DEFAULT_CONFIG = MockServerConfig(fast=True)

    def __init__(
        self,
        model_endpoint: ModelEndpointInfo,
        config: MockServerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FakeTransport.

        Args:
            model_endpoint: Model endpoint configuration.
            config: Optional MockServerConfig for test isolation.
                Defaults to zero-latency config for fast testing.
        """
        super().__init__(model_endpoint=model_endpoint, **kwargs)
        self.config = config or self._DEFAULT_CONFIG
        self.warning(
            "*** Using FakeTransport to bypass HTTP. This is for component integration testing only. ***"
        )

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return transport metadata for discovery and registration."""
        return TransportMetadata(
            transport_type=TransportType.HTTP, url_schemes=["http", "https"]
        )

    def get_url(self, request_info: RequestInfo) -> str:
        """Return fake URL (not actually used since we bypass HTTP)."""
        return self.model_endpoint.endpoint.base_url

    # =========================================================================
    # Helper methods to reduce duplication
    # =========================================================================

    def _parse_payload(
        self, payload: RequestInputT, model_class: type[RequestInputT]
    ) -> RequestInputT:
        """Parse payload into a Pydantic model."""
        if isinstance(payload, dict):
            return model_class.model_validate(payload)
        return model_class.model_validate_json(payload)

    def _make_json_record(
        self, start_perf_ns: int, response_data: dict[str, Any]
    ) -> RequestRecord:
        """Create a RequestRecord with a JSON response."""
        end_perf_ns = time.perf_counter_ns()
        return RequestRecord(
            start_perf_ns=start_perf_ns,
            end_perf_ns=end_perf_ns,
            timestamp_ns=time.time_ns(),
            status=200,
            responses=[
                TextResponse(
                    perf_ns=end_perf_ns,
                    content_type="application/json",
                    text=orjson.dumps(response_data).decode("utf-8"),
                )
            ],
        )

    async def _stream_to_record(
        self,
        stream: AsyncGenerator[bytes, None],
        start_perf_ns: int,
        first_token_callback: FirstTokenCallback | None,
    ) -> RequestRecord:
        """Consume an SSE stream and build a RequestRecord."""
        responses: list[SSEMessage] = []
        first_token_fired = False

        async for chunk_bytes in stream:
            perf_ns = time.perf_counter_ns()
            message = SSEMessage.parse(chunk_bytes.rstrip(b"\n"), perf_ns)
            responses.append(message)

            if first_token_callback and not first_token_fired:
                ttft_ns = perf_ns - start_perf_ns
                if await first_token_callback(ttft_ns, message):
                    first_token_fired = True

            # IMPORTANT! This is critical to prevent tight loops from starving the event loop.
            # Without this yield, only a single request would be able to run at a time.
            await yield_to_event_loop()

        return RequestRecord(
            start_perf_ns=start_perf_ns,
            end_perf_ns=time.perf_counter_ns(),
            timestamp_ns=time.time_ns(),
            status=200,
            responses=responses,
        )

    # =========================================================================
    # Request dispatch
    # =========================================================================

    async def _dispatch(
        self,
        payload: RequestInputT,
        endpoint_type: EndpointType,
        request_class: type[RequestInputT],
        handler: HandlerFn,
        *,
        stream_fn: StreamFn | None = None,
        build_response: BuildResponseFn | None = None,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Parse request, create context, and dispatch to handler."""
        start_perf_ns = time.perf_counter_ns()
        endpoint_path = plugins.get_endpoint_metadata(endpoint_type).endpoint_path
        req = self._parse_payload(payload, request_class)
        ctx = make_ctx(req, endpoint_path, time.perf_counter(), self.config)
        input = HandlerInput(
            start_perf_ns=start_perf_ns,
            ctx=ctx,
            endpoint_path=endpoint_path,
            req=req,
            stream_fn=stream_fn,
            build_response=build_response,
            first_token_callback=first_token_callback,
        )
        record = await handler(input)
        if self.is_debug_enabled:
            self.debug(f"FakeTransport sent request: {record}")
        return record

    async def send_request(
        self,
        request_info: RequestInfo,
        payload: RequestInputT,
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Route request to appropriate handler based on endpoint type."""
        endpoint_type = self.model_endpoint.endpoint.type

        # Handle cancellation by running request in a task and cancelling it
        if request_info.cancel_after_ns is not None:
            cancel_delay_sec = request_info.cancel_after_ns / NANOS_PER_SECOND
            self.info(f"Request will be cancelled after {cancel_delay_sec}s")

            # Create request task
            request_task = asyncio.create_task(
                self._send_request_impl(endpoint_type, payload, first_token_callback)
            )

            # Wait for cancellation timeout or completion
            try:
                return await asyncio.wait_for(request_task, timeout=cancel_delay_sec)
            except asyncio.TimeoutError:
                # Cancel the request
                request_task.cancel()
                with suppress(asyncio.CancelledError):
                    await request_task

                # Return cancellation error
                return RequestRecord(
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns(),
                    timestamp_ns=time.time_ns(),
                    status=499,
                    error=ErrorDetails(
                        type="RequestCancellationError",
                        message=f"Request cancelled after {cancel_delay_sec:.3f}s",
                        code=499,
                    ),
                    cancellation_perf_ns=time.perf_counter_ns(),
                )

        # No cancellation - run request normally
        return await self._send_request_impl(
            endpoint_type, payload, first_token_callback
        )

    async def _send_request_impl(
        self,
        endpoint_type: EndpointType,
        payload: RequestInputT,
        first_token_callback: FirstTokenCallback | None,
    ) -> RequestRecord:
        """Internal method to actually send the request (extracted for cancellation)."""
        match endpoint_type:
            case EndpointType.CHAT:
                return await self._dispatch(
                    payload,
                    endpoint_type,
                    ChatCompletionRequest,
                    self._do_streaming,
                    stream_fn=stream_chat_completion,
                    build_response=_build_chat_response_data,
                    first_token_callback=first_token_callback,
                )
            case EndpointType.COMPLETIONS:
                return await self._dispatch(
                    payload,
                    endpoint_type,
                    CompletionRequest,
                    self._do_streaming,
                    stream_fn=stream_text_completion,
                    build_response=_build_completion_response_data,
                    first_token_callback=first_token_callback,
                )
            case EndpointType.EMBEDDINGS:
                return await self._dispatch(
                    payload,
                    endpoint_type,
                    EmbeddingRequest,
                    self._do_embedding,
                )
            case EndpointType.NIM_RANKINGS:
                return await self._dispatch(
                    payload,
                    endpoint_type,
                    RankingRequest,
                    self._do_ranking,
                    build_response=_build_nim_ranking_response_data,
                )
            case EndpointType.HF_TEI_RANKINGS:
                return await self._dispatch(
                    payload,
                    endpoint_type,
                    HFTEIRerankRequest,
                    self._do_ranking,
                    build_response=_build_hf_tei_ranking_response_data,
                )
            case EndpointType.COHERE_RANKINGS:
                return await self._dispatch(
                    payload,
                    endpoint_type,
                    CohereRerankRequest,
                    self._do_ranking,
                    build_response=_build_cohere_ranking_response_data,
                )
            case EndpointType.IMAGE_GENERATION:
                return await self._dispatch(
                    payload,
                    endpoint_type,
                    ImageGenerationRequest,
                    self._do_simple,
                    build_response=_build_image_response_data,
                )
            case EndpointType.HUGGINGFACE_GENERATE:
                return await self._dispatch(
                    payload,
                    endpoint_type,
                    TGIGenerateRequest,
                    self._do_streaming,
                    stream_fn=stream_tgi_completion,
                    build_response=_build_tgi_response_data,
                    first_token_callback=first_token_callback,
                )
            case EndpointType.SOLIDO_RAG:
                return await self._dispatch(
                    payload,
                    endpoint_type,
                    SolidoRAGRequest,
                    self._do_simple,
                    build_response=_build_solido_rag_response_data,
                )
            case _:
                raise ValueError(f"Unsupported endpoint type: {endpoint_type}")

    # =========================================================================
    # Endpoint handlers
    # =========================================================================

    async def _do_simple(self, inp: HandlerInput) -> RequestRecord:
        """Handle simple non-streaming requests (image, SOLIDO RAG)."""
        await inp.ctx.latency_sim.wait_for_tokens(len(inp.ctx.tokens))
        return self._make_json_record(
            inp.start_perf_ns, inp.build_response(inp.ctx, inp.req)
        )

    async def _do_streaming(self, inp: HandlerInput) -> RequestRecord:
        """Handle streaming completion requests (chat, text, TGI)."""
        if self.model_endpoint.endpoint.streaming:
            include_usage = getattr(inp.req, "include_usage", False)
            stream = inp.stream_fn(inp.ctx, inp.endpoint_path, include_usage)
            return await self._stream_to_record(
                stream, inp.start_perf_ns, inp.first_token_callback
            )

        await inp.ctx.latency_sim.wait_for_tokens(len(inp.ctx.tokens))
        return self._make_json_record(inp.start_perf_ns, inp.build_response(inp.ctx))

    async def _do_embedding(self, inp: HandlerInput) -> RequestRecord:
        """Handle embedding requests."""
        await _wait_for_processing(
            self.config.embedding_base_latency,
            self.config.embedding_per_input_latency,
            len(inp.req.inputs),
        )
        return self._make_json_record(
            inp.start_perf_ns, _build_embedding_response_data(inp.ctx, inp.req.inputs)
        )

    async def _do_ranking(self, inp: HandlerInput) -> RequestRecord:
        """Handle ranking requests."""
        ranked_scores = _compute_ranked_scores(
            inp.req.query_text, inp.req.passage_texts
        )
        await _wait_for_processing(
            self.config.ranking_base_latency,
            self.config.ranking_per_passage_latency,
            len(inp.req.passage_texts),
        )
        return self._make_json_record(
            inp.start_perf_ns, inp.build_response(inp.ctx, ranked_scores)
        )


# =============================================================================
# Plugin Registration - Hot-swap production implementations when imported
# =============================================================================

# Register FakeTransport for HTTP transport at max priority
plugins.register(
    PluginType.TRANSPORT,
    TransportType.HTTP,
    FakeTransport,
    priority=sys.maxsize,
    metadata={"transport_type": "http", "url_schemes": ["http", "https"]},
)
