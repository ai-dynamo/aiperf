# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import logging
import random
import time
from contextlib import asynccontextmanager

from aiperf_mock_server.config import server_config
from aiperf_mock_server.dcgm_faker import DCGMFaker
from aiperf_mock_server.models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    Ranking,
    RankingRequest,
    RankingResponse,
    TextChoice,
    TextCompletionResponse,
)
from aiperf_mock_server.utils import (
    RequestContext,
    stream_chat_completion,
    stream_text_completion,
    with_error_injection,
)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

dcgm_fakers: list[DCGMFaker] = []
logger = logging.getLogger(__name__)


def _create_dcgm_faker(seed: int | None) -> DCGMFaker:
    """Create a DCGM faker instance with current config."""
    return DCGMFaker(
        gpu_name=server_config.dcgm_gpu_name,
        num_gpus=server_config.dcgm_num_gpus,
        seed=seed,
        hostname=server_config.dcgm_hostname,
        initial_load=server_config.dcgm_initial_load,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize server on startup."""
    logger.info("Server starting: %s", server_config.model_dump())
    if server_config.random_seed is not None:
        random.seed(server_config.random_seed)

    dcgm_fakers.append(_create_dcgm_faker(server_config.dcgm_seed))
    dcgm_fakers.append(
        _create_dcgm_faker(
            None if server_config.dcgm_seed is None else server_config.dcgm_seed + 1
        )
    )

    logger.info(
        "DCGM faker initialized with %d %s GPUs",
        server_config.dcgm_num_gpus,
        server_config.dcgm_gpu_name,
    )
    yield


app = FastAPI(title="AIPerf Mock Server", version="2.0.0", lifespan=lifespan)

# ============================================================================
# Chat Completions
# ============================================================================


@app.post("/v1/chat/completions", response_model=None)
@with_error_injection
async def chat_completions(
    req: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """Chat completion endpoint."""
    ctx = RequestContext(req)

    if req.stream:
        return StreamingResponse(
            stream_chat_completion(ctx),
            media_type="text/event-stream",
        )

    await ctx.wait_until_completion()

    return ChatCompletionResponse(
        id=ctx.request_id,
        created=int(time.time()),
        model=ctx.request.model,
        choices=[
            ChatChoice(
                index=0,
                finish_reason=ctx.tokenized.finish_reason,
                message=ChatMessage(
                    role="assistant",
                    content=ctx.tokenized.content,
                    reasoning_content=ctx.tokenized.reasoning_content,
                ),
            )
        ],
        usage=ctx.tokenized.create_usage(),
    )


# ============================================================================
# Text Completions
# ============================================================================


@app.post("/v1/completions", response_model=None)
@with_error_injection
async def completions(
    req: CompletionRequest,
) -> TextCompletionResponse | StreamingResponse:
    """Text completion endpoint."""
    ctx = RequestContext(req)

    if req.stream:
        return StreamingResponse(
            stream_text_completion(ctx),
            media_type="text/event-stream",
        )

    await ctx.wait_until_completion()

    choice = TextChoice(
        index=0,
        finish_reason=ctx.tokenized.finish_reason,
        text=ctx.tokenized.content,
    )

    return TextCompletionResponse(
        id=ctx.request_id,
        created=int(time.time()),
        model=ctx.request.model,
        choices=[choice],
        usage=ctx.tokenized.create_usage(),
    )


# ============================================================================
# Embeddings
# ============================================================================


@app.post("/v1/embeddings", response_model=None)
@with_error_injection
async def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    """Embedding endpoint."""
    ctx = RequestContext(req)
    await ctx.wait_until_completion()

    def generate_embedding(text: str) -> list[float]:
        """Generate deterministic embedding from text using stable hash."""
        digest = hashlib.blake2s(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest, byteorder="big")
        rng = random.Random(seed)
        return [rng.random() - 0.5 for _ in range(768)]

    return EmbeddingResponse(
        id=ctx.request_id,
        created=int(time.time()),
        model=ctx.request.model,
        data=[
            Embedding(
                index=i,
                embedding=generate_embedding(text),
            )
            for i, text in enumerate(req.inputs)
        ],
        usage=ctx.tokenized.create_usage(),
    )


# ============================================================================
# Rankings
# ============================================================================


def _compute_mock_score(query: str, passage: str) -> float:
    """Compute deterministic mock relevance score for all ranking mocks."""
    combined = f"{query}|{passage}"
    digest = hashlib.blake2s(combined.encode("utf-8")).digest()
    int_digest = int.from_bytes(digest, byteorder="big")
    return (int_digest % 1000) / 1000.0


# ============================================================================
# NIM Rankings Endpoint
# ============================================================================


@app.post("/v1/ranking", response_model=None)
@with_error_injection
async def rankings(req: RankingRequest) -> RankingResponse:
    """Mock NVIDIA NIM /v1/ranking endpoint."""
    ctx = RequestContext(req)

    rankings = sorted(
        [
            Ranking(
                index=i,
                relevance_score=_compute_mock_score(req.query_text, text),
            )
            for i, text in enumerate(req.passage_texts)
        ],
        key=lambda x: x.relevance_score,
        reverse=True,
    )

    await ctx.wait_until_completion()
    return RankingResponse(
        id=ctx.request_id,
        model=req.model,
        rankings=rankings,
        usage=ctx.tokenized.create_usage(),
    )


# ============================================================================
# HuggingFace TEI Rankings Endpoint
# ============================================================================


@app.post("/rerank", response_model=None)
@with_error_injection
async def hf_tei_rerank(req: dict) -> dict:
    """Mock HuggingFace TEI /rerank endpoint."""
    query = req.get("query", "")
    passages = req.get("texts") or req.get("documents") or []

    results = [
        {"index": i, "score": _compute_mock_score(query, p)}
        for i, p in enumerate(passages)
    ]
    results.sort(key=lambda r: r["score"], reverse=True)

    return {"results": results}


# ============================================================================
# Cohere Rankings Endpoint
# ============================================================================


@app.post("/v2/rerank", response_model=None)
@with_error_injection
async def cohere_rerank(req: dict) -> dict:
    """Mock Cohere /v2/rerank endpoint."""
    query = req.get("query", "")
    passages = req.get("documents") or []

    results = [
        {"index": i, "relevance_score": _compute_mock_score(query, p)}
        for i, p in enumerate(passages)
    ]
    results.sort(key=lambda r: r["relevance_score"], reverse=True)

    return {"results": results}


# ============================================================================
# Custom Multimodal Endpoint
# ============================================================================


@app.post("/v1/custom-multimodal", response_model=None)
@with_error_injection
async def custom_multimodal(req: dict) -> dict:
    """Mock endpoint with custom multi-modal format.

    Expected format:
    {
        "modality_bundle": {
            "text_fragments": ["text1", "text2"],
            "visual_assets": {
                "images": ["base64..."],
                "videos": ["base64..."]
            },
            "audio_streams": ["base64..."]
        },
        "inference_params": {
            "model_id": "...",
            "sampling_config": {...}
        }
    }

    Returns format:
    {
        "completion": {
            "generated_text": "...",
            "metadata": {
                "tokens_used": {...}
            }
        }
    }
    """
    try:
        # Extract the multimodal bundle
        bundle = req.get("modality_bundle", {})
        text_fragments = bundle.get("text_fragments", [])
        visual_assets = bundle.get("visual_assets", {})
        images = visual_assets.get("images", [])
        videos = visual_assets.get("videos", [])
        audio_streams = bundle.get("audio_streams", [])

        # Extract inference params
        inference_params = req.get("inference_params", {})
        model_id = inference_params.get("model_id", "default-model")

        # Create a mock request for timing - use a simple valid ChatCompletionRequest
        text_content = " ".join(text_fragments) if text_fragments else "default text"
        mock_req = ChatCompletionRequest(
            model=model_id or "default-model",
            messages=[{"role": "user", "content": text_content}],
        )
        ctx = RequestContext(mock_req)
        await ctx.wait_until_completion()

        # Build response with custom format
        response_text = f"Processed {len(text_fragments)} text fragments"
        if images:
            response_text += f", {len(images)} images"
        if videos:
            response_text += f", {len(videos)} videos"
        if audio_streams:
            response_text += f", {len(audio_streams)} audio streams"

        usage = ctx.tokenized.create_usage()
        return {
            "text": response_text,  # Use "text" field for auto-detection
            "completion": {
                "generated_text": response_text,
                "metadata": {
                    "tokens_used": {
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens,
                    }
                },
            },
        }
    except Exception as e:
        logger.error(f"Error in custom_multimodal endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# HuggingFace Generate Endpoint
# ============================================================================


@app.post("/generate", response_model=None)
@with_error_injection
async def huggingface_generate(req: dict):
    """Mock HuggingFace TGI /generate endpoint (non-streaming)."""
    prompt = req.get("inputs") or req.get("prompt") or "Hello!"
    max_new_tokens = req.get("parameters", {}).get("max_new_tokens", 50)

    fake_text = f"{prompt} [mocked generation, {max_new_tokens} tokens]"
    return JSONResponse(content={"generated_text": fake_text})


@app.post("/generate_stream", response_model=None)
@with_error_injection
async def huggingface_generate_stream(req: dict):
    """Mock HuggingFace TGI /generate_stream endpoint (streaming)."""
    prompt = req.get("inputs") or req.get("prompt") or "Hello!"
    max_new_tokens = req.get("parameters", {}).get("max_new_tokens", 10)

    async def event_stream():
        for i in range(max_new_tokens):
            yield f'data: {{"token": {{"text": " token_{i}"}}}}\n\n'
            await asyncio.sleep(0.001)
        yield f'data: {{"generated_text": "{prompt} [mocked streaming done]"}}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
