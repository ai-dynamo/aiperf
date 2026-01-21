# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib
import logging
import random
import time
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

import orjson
from aiperf_mock_server.config import server_config
from aiperf_mock_server.dcgm_faker import DCGMFaker
from aiperf_mock_server.metrics import (
    AIPERF_MOCK_REGISTRY,
    DYNAMO_DECODE_REGISTRY,
    DYNAMO_FRONTEND_REGISTRY,
    DYNAMO_PREFILL_REGISTRY,
    SERVER_UPTIME_SECONDS,
    SGLANG_REGISTRY,
    STREAMING_REQUESTS_TOTAL,
    TRTLLM_REGISTRY,
    VLLM_REGISTRY,
)
from aiperf_mock_server.metrics_utils import (
    async_track_llm_request,
    async_track_request,
    init_model_config,
    record_embedding_success,
    record_ranking_success,
    record_request_bytes,
    record_tgi_success,
    register_dcgm_load_callback,
    track_llm_request,
    track_request,
)
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
    with_error_injection,
)
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import ORJSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from starlette.requests import Request

dcgm_fakers: list[DCGMFaker] = []
server_start_time: float = 0.0
logger = logging.getLogger(__name__)


def metrics_response(registry: CollectorRegistry) -> Response:
    """Generate a Prometheus metrics response from a registry."""
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


def _create_dcgm_faker(seed: int | None) -> DCGMFaker:
    """Create a DCGM faker instance with current config."""
    return DCGMFaker(
        gpu_name=server_config.dcgm_gpu_name,
        num_gpus=server_config.dcgm_num_gpus,
        seed=seed,
        hostname=server_config.dcgm_hostname,
    )


def _update_dcgm_load(load: float) -> None:
    """Update load on all DCGM fakers based on inflight requests."""
    for faker in dcgm_fakers:
        faker.set_load(load)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize server on startup."""
    global server_start_time
    server_start_time = time.time()
    logger.info("Server starting: %s", server_config.model_dump())
    if server_config.random_seed is not None:
        random.seed(server_config.random_seed)

    if not server_config.no_tokenizer:
        from aiperf_mock_server.tokens import _load_corpus

        _load_corpus()

    dcgm_fakers.append(_create_dcgm_faker(server_config.dcgm_seed))
    dcgm_fakers.append(
        _create_dcgm_faker(
            None if server_config.dcgm_seed is None else server_config.dcgm_seed + 1
        )
    )

    # Register callback to update DCGM load based on token throughput (auto-scaling)
    if server_config.dcgm_auto_load:
        register_dcgm_load_callback(
            _update_dcgm_load,
            server_config.dcgm_min_throughput,
            server_config.dcgm_window_sec,
        )

    if server_config.dcgm_auto_load:
        logger.info(
            "DCGM faker initialized with %d %s GPUs (auto-load enabled, %.1fs window)",
            server_config.dcgm_num_gpus,
            server_config.dcgm_gpu_name,
            server_config.dcgm_window_sec,
        )
    else:
        logger.info(
            "DCGM faker initialized with %d %s GPUs (auto-load disabled)",
            server_config.dcgm_num_gpus,
            server_config.dcgm_gpu_name,
        )
    yield


app = FastAPI(title="AIPerf Mock Server", version="2.0.0", lifespan=lifespan)


class TimingMiddleware:
    """Pure ASGI middleware - captures timing before ANY processing."""

    def __init__(self, inner_app):
        self.app = inner_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Initialize state dict if not present (e.g., in test environments)
            if "state" not in scope:
                scope["state"] = {}
            scope["state"]["start_time"] = perf_counter()
        await self.app(scope, receive, send)


# Wrap FastAPI with ASGI middleware for earliest possible timing
asgi_app = TimingMiddleware(app)


# ============================================================================
# Chat Completions
# ============================================================================


def _build_chat_response_data(ctx: RequestCtx) -> dict[str, Any]:
    """Build non-streaming chat completion response data."""
    message: dict[str, Any] = {"role": "assistant", "content": ctx.content}
    if ctx.reasoning_content:
        message["reasoning_content"] = ctx.reasoning_content
    return {
        "id": ctx.request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": ctx.model,
        "choices": [
            {"index": 0, "finish_reason": ctx.finish_reason, "message": message}
        ],
        "usage": ctx.usage,
    }


@app.post("/v1/chat/completions", response_model=None)
@with_error_injection
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
) -> ORJSONResponse | StreamingResponse:
    """Chat completion endpoint."""
    endpoint = "/v1/chat/completions"
    init_model_config(req.model)
    ctx = make_ctx(req, endpoint, request.state.start_time)

    if req.stream:
        STREAMING_REQUESTS_TOTAL.labels(endpoint=endpoint, model=req.model).inc()
        return StreamingResponse(
            _chat_stream_wrapper(ctx, req, endpoint),
            media_type="text/event-stream",
        )

    with track_llm_request(ctx, req.model, endpoint):
        await ctx.latency_sim.wait_for_tokens(len(ctx.tokens))
        response_data = _build_chat_response_data(ctx)
        response_bytes = len(orjson.dumps(response_data))
        record_request_bytes(endpoint, len(ctx.tokenized.text), response_bytes)
        return ORJSONResponse(response_data)


async def _chat_stream_wrapper(
    ctx: RequestCtx, req: ChatCompletionRequest, endpoint: str
):
    """Wrapper for streaming that records metrics after completion."""
    async with async_track_llm_request(ctx, req.model, endpoint):
        async for chunk in stream_chat_completion(ctx, endpoint, req.include_usage):
            yield chunk


# ============================================================================
# Text Completions
# ============================================================================


def _build_completion_response_data(ctx: RequestCtx) -> dict[str, Any]:
    """Build non-streaming text completion response data."""
    return {
        "id": ctx.request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": ctx.model,
        "choices": [
            {
                "index": 0,
                "finish_reason": ctx.finish_reason,
                "text": ctx.content,
            }
        ],
        "usage": ctx.usage,
    }


@app.post("/v1/completions", response_model=None)
@with_error_injection
async def completions(
    req: CompletionRequest,
    request: Request,
) -> ORJSONResponse | StreamingResponse:
    """Text completion endpoint."""
    endpoint = "/v1/completions"
    init_model_config(req.model)
    ctx = make_ctx(req, endpoint, request.state.start_time)

    if req.stream:
        STREAMING_REQUESTS_TOTAL.labels(endpoint=endpoint, model=req.model).inc()
        return StreamingResponse(
            _text_stream_wrapper(ctx, req, endpoint),
            media_type="text/event-stream",
        )

    with track_llm_request(ctx, req.model, endpoint):
        await ctx.latency_sim.wait_for_tokens(len(ctx.tokens))
        response_data = _build_completion_response_data(ctx)
        response_bytes = len(orjson.dumps(response_data))
        record_request_bytes(endpoint, len(ctx.tokenized.text), response_bytes)
        return ORJSONResponse(response_data)


async def _text_stream_wrapper(ctx: RequestCtx, req: CompletionRequest, endpoint: str):
    """Wrapper for text streaming that records metrics after completion."""
    async with async_track_llm_request(ctx, req.model, endpoint):
        async for chunk in stream_text_completion(ctx, endpoint, req.include_usage):
            yield chunk


# ============================================================================
# Embeddings
# ============================================================================


async def _wait_for_processing(base_ms: float, per_unit_ms: float, units: int) -> None:
    """Wait for processing based on base latency + per-unit latency."""
    total_ms = base_ms + (per_unit_ms * units)
    if total_ms > 0:
        await asyncio.sleep(total_ms / 1000.0)


def generate_embedding(text: str, dim: int = 768) -> list[float]:
    """Generate deterministic embedding from text using stable hash.

    Args:
        text: Input text to generate embedding for.
        dim: Embedding dimension (default 768).

    Returns:
        List of floats representing the embedding vector.
    """
    digest = hashlib.blake2s(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest, byteorder="big")
    rng = random.Random(seed)
    return [rng.random() - 0.5 for _ in range(dim)]


def _build_embedding_response_data(
    ctx: RequestCtx, inputs: list[str]
) -> dict[str, Any]:
    """Build embedding response data."""
    return {
        "object": "list",
        "model": ctx.model,
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": generate_embedding(text),
            }
            for i, text in enumerate(inputs)
        ],
        "usage": ctx.usage,
    }


@app.post("/v1/embeddings", response_model=None)
@with_error_injection
async def embeddings(req: EmbeddingRequest, request: Request) -> ORJSONResponse:
    """Embedding endpoint."""
    endpoint = "/v1/embeddings"
    start_time = request.state.start_time
    ctx = make_ctx(req, endpoint, start_time)

    with track_request(endpoint, req.model):
        await _wait_for_processing(
            server_config.embedding_base_latency,
            server_config.embedding_per_input_latency,
            len(req.inputs),
        )

        record_embedding_success(
            endpoint,
            req.model,
            ctx.usage["prompt_tokens"],
            len(req.inputs),
            perf_counter() - start_time,
        )

        return ORJSONResponse(_build_embedding_response_data(ctx, req.inputs))


# ============================================================================
# Rankings
# ============================================================================


def _compute_mock_score(query: str, passage: str) -> float:
    """Compute deterministic mock relevance score for all ranking mocks."""
    combined = f"{query}|{passage}"
    digest = hashlib.blake2s(combined.encode("utf-8")).digest()
    int_digest = int.from_bytes(digest, byteorder="big")
    return (int_digest % 1000) / 1000.0


def _compute_ranked_scores(query: str, passages: list[str]) -> list[tuple[int, float]]:
    """Compute and sort mock scores for passages, returning (index, score) pairs."""
    scores = [(i, _compute_mock_score(query, p)) for i, p in enumerate(passages)]
    return sorted(scores, key=lambda x: x[1], reverse=True)


RankingRequestT = RankingRequest | HFTEIRerankRequest | CohereRerankRequest


async def _handle_ranking_request(
    req: RankingRequestT, endpoint: str
) -> tuple[RequestCtx, list[tuple[int, float]]]:
    """Common ranking request handler. Returns context and sorted (index, score) pairs."""
    start_time = perf_counter()
    ctx = make_ctx(req, endpoint, start_time)

    with track_request(endpoint, req.model):
        ranked_scores = _compute_ranked_scores(req.query_text, req.passage_texts)

        await _wait_for_processing(
            server_config.ranking_base_latency,
            server_config.ranking_per_passage_latency,
            len(req.passage_texts),
        )

        record_ranking_success(
            endpoint,
            req.model,
            ctx.usage["prompt_tokens"],
            len(req.passage_texts),
            perf_counter() - start_time,
        )
        return ctx, ranked_scores


# ============================================================================
# NIM Rankings Endpoint
# ============================================================================


def _build_nim_ranking_response_data(
    ctx: RequestCtx, ranked_scores: list[tuple[int, float]]
) -> dict[str, Any]:
    """Build NIM /v1/ranking response data."""
    return {
        "id": ctx.request_id,
        "object": "rankings",
        "model": ctx.model,
        "rankings": [{"index": i, "relevance_score": s} for i, s in ranked_scores],
        "usage": ctx.usage,
    }


@app.post("/v1/ranking", response_model=None)
@with_error_injection
async def rankings(req: RankingRequest) -> ORJSONResponse:
    """Mock NVIDIA NIM /v1/ranking endpoint."""
    ctx, ranked_scores = await _handle_ranking_request(req, "/v1/ranking")
    return ORJSONResponse(_build_nim_ranking_response_data(ctx, ranked_scores))


# ============================================================================
# HuggingFace TEI Rankings Endpoint
# ============================================================================


def _build_hf_tei_ranking_response_data(
    _ctx: RequestCtx, ranked_scores: list[tuple[int, float]]
) -> dict[str, Any]:
    """Build HuggingFace TEI /rerank response data."""
    return {"results": [{"index": i, "score": s} for i, s in ranked_scores]}


@app.post("/rerank", response_model=None)
@with_error_injection
async def hf_tei_rerank(req: HFTEIRerankRequest) -> ORJSONResponse:
    """Mock HuggingFace TEI /rerank endpoint."""
    ctx, ranked_scores = await _handle_ranking_request(req, "/rerank")
    return ORJSONResponse(_build_hf_tei_ranking_response_data(ctx, ranked_scores))


# ============================================================================
# Cohere Rankings Endpoint
# ============================================================================


def _build_cohere_ranking_response_data(
    _ctx: RequestCtx, ranked_scores: list[tuple[int, float]]
) -> dict[str, Any]:
    """Build Cohere /v2/rerank response data."""
    return {"results": [{"index": i, "relevance_score": s} for i, s in ranked_scores]}


@app.post("/v2/rerank", response_model=None)
@with_error_injection
async def cohere_rerank(req: CohereRerankRequest) -> ORJSONResponse:
    """Mock Cohere /v2/rerank endpoint."""
    ctx, ranked_scores = await _handle_ranking_request(req, "/v2/rerank")
    return ORJSONResponse(_build_cohere_ranking_response_data(ctx, ranked_scores))


# ============================================================================
# Custom Multimodal Endpoint
# ============================================================================


@app.post("/v1/custom-multimodal", response_model=None)
@with_error_injection
async def custom_multimodal(req: dict, request: Request) -> dict:
    """Mock endpoint with custom multi-modal format."""
    endpoint = "/v1/custom-multimodal"
    inference_params = req.get("inference_params", {})
    model_id = inference_params.get("model_id", "default-model")

    # Parse multimodal input
    bundle = req.get("modality_bundle", {})
    text_fragments = bundle.get("text_fragments", [])
    visual_assets = bundle.get("visual_assets", {})
    images = visual_assets.get("images", [])
    videos = visual_assets.get("videos", [])
    audio_streams = bundle.get("audio_streams", [])

    # Create mock chat request for LLM simulation
    text_content = " ".join(text_fragments) if text_fragments else "default text"
    mock_req = ChatCompletionRequest(
        model=model_id or "default-model",
        messages=[{"role": "user", "content": text_content}],
    )
    ctx = make_ctx(mock_req, endpoint, request.state.start_time)

    with track_llm_request(ctx, mock_req.model, endpoint):
        await ctx.latency_sim.wait_for_tokens(len(ctx.tokens))

        response_text = f"Processed {len(text_fragments)} text fragments"
        if images:
            response_text += f", {len(images)} images"
        if videos:
            response_text += f", {len(videos)} videos"
        if audio_streams:
            response_text += f", {len(audio_streams)} audio streams"

        return {
            "text": response_text,
            "completion": {
                "generated_text": response_text,
                "metadata": {
                    "tokens_used": {
                        "input": ctx.usage["prompt_tokens"],
                        "output": ctx.usage["completion_tokens"],
                        "total": ctx.usage["total_tokens"],
                    }
                },
            },
        }


# ============================================================================
# HuggingFace Generate Endpoint
# ============================================================================


def _build_tgi_response_data(ctx: RequestCtx) -> dict[str, Any]:
    """Build non-streaming TGI /generate response."""
    return {"generated_text": ctx.content}


@app.post("/generate", response_model=None)
@with_error_injection
async def huggingface_generate(
    req: TGIGenerateRequest, request: Request
) -> ORJSONResponse:
    """Mock HuggingFace TGI /generate endpoint (non-streaming)."""
    endpoint = "/generate"
    start_time = request.state.start_time
    ctx = make_ctx(req, endpoint, start_time)

    with track_request(endpoint, req.model):
        await ctx.latency_sim.wait_for_tokens(len(ctx.tokens))
        record_tgi_success(endpoint, ctx.usage, perf_counter() - start_time)
        return ORJSONResponse(_build_tgi_response_data(ctx))


@app.post("/generate_stream", response_model=None)
@with_error_injection
async def huggingface_generate_stream(req: TGIGenerateRequest, request: Request):
    """Mock HuggingFace TGI /generate_stream endpoint (streaming)."""
    endpoint = "/generate_stream"
    start_time = request.state.start_time
    ctx = make_ctx(req, endpoint, start_time)
    STREAMING_REQUESTS_TOTAL.labels(endpoint=endpoint, model=req.model).inc()
    return StreamingResponse(
        _tgi_stream_wrapper(ctx, endpoint, start_time),
        media_type="text/event-stream",
    )


async def _tgi_stream_wrapper(ctx: RequestCtx, endpoint: str, start_time: float):
    """Wrapper for TGI streaming that records metrics after completion."""
    async with async_track_request(endpoint, ctx.model):
        async for chunk in stream_tgi_completion(ctx, endpoint):
            yield chunk
        record_tgi_success(endpoint, ctx.usage, perf_counter() - start_time)


# ============================================================================
# Image Generation
# ============================================================================


def _generate_mock_jpeg_b64(prompt: str, index: int = 0) -> str:
    """Generate deterministic mock base64 JPEG image from prompt.

    Creates a minimal valid JPEG file that can be decoded by standard
    image libraries. The image content is deterministically generated
    based on the prompt.
    """
    combined = f"{prompt}|{index}"
    digest = hashlib.blake2s(combined.encode("utf-8")).digest()

    # Create a minimal valid JPEG (1x1 pixel)
    jpeg_data = b"\xff\xd8"  # SOI (Start of Image)
    jpeg_data += b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"  # JFIF
    jpeg_data += b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"  # SOF0
    jpeg_data += (
        b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x09"
    )  # DHT
    jpeg_data += (
        b"\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00"
    )  # DHT
    jpeg_data += b"\xff\xdb\x00\x43\x00" + digest[:64]  # DQT (deterministic)
    jpeg_data += b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00" + digest[64:80]  # SOS
    jpeg_data += b"\xff\xd9"  # EOI (End of Image)

    return base64.b64encode(jpeg_data).decode("utf-8")


def _build_image_response_data(
    ctx: RequestCtx, req: ImageGenerationRequest
) -> dict[str, Any]:
    """Build non-streaming image generation response."""
    data = []
    for i in range(req.n):
        image_data: dict[str, Any] = {
            "b64_json": _generate_mock_jpeg_b64(req.prompt, i)
        }
        if req.response_format == "url":
            image_data["url"] = f"https://mock.image.url/{i}"
        data.append(image_data)

    response_data: dict[str, Any] = {"created": int(time.time()), "data": data}

    if req.size:
        response_data["size"] = req.size
    if req.quality:
        response_data["quality"] = req.quality
    if req.style:
        response_data["style"] = req.style

    response_data["usage"] = ctx.usage
    return response_data


@app.post("/v1/images/generations", response_model=None)
@with_error_injection
async def image_generation(
    req: ImageGenerationRequest, request: Request
) -> ORJSONResponse | StreamingResponse:
    """Mock OpenAI Image Generation endpoint.

    Supports both streaming and non-streaming responses.
    Returns deterministic base64-encoded JPEG images.
    """
    endpoint = "/v1/images/generations"
    start_time = request.state.start_time
    mock_req = ChatCompletionRequest(
        model=req.model, messages=[{"role": "user", "content": req.prompt}]
    )
    ctx = make_ctx(mock_req, endpoint, start_time)

    if req.stream:
        STREAMING_REQUESTS_TOTAL.labels(endpoint=endpoint, model=req.model).inc()

        async def image_stream():
            async with async_track_llm_request(ctx, req.model, endpoint):
                for i in range(req.n):
                    await ctx.latency_sim.wait_for_tokens(len(ctx.tokens) // req.n)
                    chunk = {
                        "b64_json": _generate_mock_jpeg_b64(req.prompt, i),
                        "partial_image_index": i,
                    }
                    if req.size:
                        chunk["size"] = req.size
                    if req.quality:
                        chunk["quality"] = req.quality
                    yield b"data: " + orjson.dumps(chunk) + b"\n\n"
                yield b"data: [DONE]\n\n"

        return StreamingResponse(image_stream(), media_type="text/event-stream")

    with track_llm_request(ctx, req.model, endpoint):
        await ctx.latency_sim.wait_for_tokens(len(ctx.tokens))
        return ORJSONResponse(_build_image_response_data(ctx, req))


# ============================================================================
# SOLIDO RAG
# ============================================================================


def _build_solido_rag_response_data(
    ctx: RequestCtx, req: SolidoRAGRequest
) -> dict[str, Any]:
    """Build SOLIDO RAG response data."""
    query_text = " ".join(req.query)
    sources = []
    num_sources = min(3, len(req.query))
    for i in range(num_sources):
        source_hash = hashlib.blake2s(f"{query_text}|source{i}".encode()).hexdigest()[
            :8
        ]
        sources.append(
            {
                "id": f"doc_{source_hash}",
                "title": f"Document {i + 1}",
                "score": 0.9 - (i * 0.1),
                "content": f"Source content for query: {query_text[:50]}...",
            }
        )
    return {
        "content": ctx.content,
        "sources": sources,
        "filters": req.filters,
        "inference_model": req.inference_model,
    }


@app.post("/rag/api/prompt", response_model=None)
@with_error_injection
async def solido_rag(req: SolidoRAGRequest, request: Request) -> ORJSONResponse:
    """Mock SOLIDO RAG endpoint.

    Processes RAG queries with filters and returns generated content with sources.
    """
    endpoint = "/rag/api/prompt"
    start_time = request.state.start_time

    # Create mock chat request for token counting and timing
    query_text = " ".join(req.query)
    mock_req = ChatCompletionRequest(
        model=req.inference_model,
        messages=[{"role": "user", "content": query_text}],
    )
    ctx = make_ctx(mock_req, endpoint, start_time)

    with track_llm_request(ctx, req.inference_model, endpoint):
        await ctx.latency_sim.wait_for_tokens(len(ctx.tokens))
        return ORJSONResponse(_build_solido_rag_response_data(ctx, req))


# ============================================================================
# Health & Info
# ============================================================================


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "config": server_config.model_dump()}


@app.get("/")
async def root():
    """Root info."""
    return {
        "message": "AIPerf Mock Server",
        "version": "2.0.0",
        "config": server_config.model_dump(),
    }


# ============================================================================
# DCGM Metrics
# ============================================================================


@app.get("/dcgm{instance_id:int}/metrics")
async def dcgm_metrics(instance_id: int) -> PlainTextResponse:
    """DCGM metrics endpoint (Prometheus format)."""
    index = instance_id - 1
    if index < 0 or index >= len(dcgm_fakers):
        raise HTTPException(status_code=404, detail="Invalid DCGM instance")
    return PlainTextResponse(dcgm_fakers[index].generate(), media_type="text/plain")


# ============================================================================
# Prometheus Metrics Endpoints
# ============================================================================


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """AIPerf mock server Prometheus metrics endpoint.

    Returns AIPerf mock server specific metrics:
    - Request counts by endpoint, method, and status
    - Request latency histograms
    - Token counts (prompt/completion) by endpoint and model
    - Streaming metrics (tokens streamed, TTFT, ITL)
    - In-flight request gauges
    - Error counts by type
    - Server uptime
    """
    # Update uptime on each scrape
    if server_start_time > 0:
        SERVER_UPTIME_SECONDS.set(time.time() - server_start_time)
    return metrics_response(AIPERF_MOCK_REGISTRY)


@app.get("/vllm/metrics")
async def vllm_metrics() -> Response:
    """vLLM-compatible Prometheus metrics endpoint.

    Returns metrics matching vLLM server format:
    - vllm:e2e_request_latency_seconds
    - vllm:time_to_first_token_seconds
    - vllm:inter_token_latency_seconds
    - vllm:prompt_tokens, vllm:generation_tokens
    - vllm:num_requests_running, vllm:num_requests_waiting
    - vllm:request_success, vllm:request_queue_time_seconds
    """
    return metrics_response(VLLM_REGISTRY)


@app.get("/sglang/metrics")
async def sglang_metrics() -> Response:
    """SGLang-compatible Prometheus metrics endpoint.

    Returns metrics matching SGLang server format:
    - sglang:e2e_request_latency_seconds
    - sglang:time_to_first_token_seconds
    - sglang:queue_time_seconds
    - sglang:num_running_reqs, sglang:num_queue_reqs
    - sglang:gen_throughput, sglang:cache_hit_rate
    """
    return metrics_response(SGLANG_REGISTRY)


@app.get("/trtllm/metrics")
async def trtllm_metrics() -> Response:
    """TensorRT-LLM-compatible Prometheus metrics endpoint.

    Returns metrics matching TensorRT-LLM server format:
    - trtllm:e2e_request_latency_seconds
    - trtllm:time_to_first_token_seconds
    - trtllm:time_per_output_token_seconds
    - trtllm:request_queue_time_seconds
    - trtllm:request_success
    """
    return metrics_response(TRTLLM_REGISTRY)


@app.get("/dynamo_frontend/metrics")
async def dynamo_frontend_metrics() -> Response:
    """Dynamo frontend Prometheus metrics endpoint.

    Returns metrics matching Dynamo frontend format:
    - dynamo_frontend_request_duration_seconds
    - dynamo_frontend_time_to_first_token_seconds
    - dynamo_frontend_inter_token_latency_seconds
    - dynamo_frontend_requests
    - dynamo_frontend_input_sequence_tokens, dynamo_frontend_output_tokens
    - dynamo_frontend_queued_requests, dynamo_frontend_inflight_requests
    """
    return metrics_response(DYNAMO_FRONTEND_REGISTRY)


@app.get("/dynamo_component/prefill/metrics")
async def dynamo_prefill_metrics() -> Response:
    """Dynamo prefill worker Prometheus metrics endpoint.

    Returns metrics matching Dynamo component format for prefill workers:
    - dynamo_component_request_duration_seconds
    - dynamo_component_requests
    - dynamo_component_inflight_requests
    - dynamo_component_kvstats_* (KV cache stats)
    """
    return metrics_response(DYNAMO_PREFILL_REGISTRY)


@app.get("/dynamo_component/decode/metrics")
async def dynamo_decode_metrics() -> Response:
    """Dynamo decode worker Prometheus metrics endpoint.

    Returns metrics matching Dynamo component format for decode workers:
    - dynamo_component_request_duration_seconds
    - dynamo_component_requests
    - dynamo_component_inflight_requests
    - dynamo_component_kvstats_* (KV cache stats)
    """
    return metrics_response(DYNAMO_DECODE_REGISTRY)
