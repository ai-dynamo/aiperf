# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI server for integration testing with configurable latencies."""

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .config import MockServerConfig
from .models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ConfigureMessage,
    Role,
    Usage,
)
from .tokenizer_service import tokenizer_service

logger = logging.getLogger(__name__)

# Global server configuration
server_config: MockServerConfig = MockServerConfig()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize tokenizers and other startup tasks."""
    logger.info("Server configuration: %s", server_config.model_dump())

    tokenizer_models = [
        *server_config.tokenizer_models,
        server_config.fallback_tokenizer,
    ]

    logger.info(f"Pre-loading tokenizer models: {tokenizer_models}")
    tokenizer_service.set_fallback_tokenizer(server_config.fallback_tokenizer)
    tokenizer_service.load_tokenizers(tokenizer_models)
    logger.info("Tokenizer models loaded successfully")

    yield


app = FastAPI(
    title="AIPerf Integration Test Server",
    description="FastAPI server that echoes prompts token by token with configurable latencies",
    version="1.0.0",
    lifespan=lifespan,
)


def set_server_config(config: MockServerConfig) -> None:
    """Set the global server configuration."""
    global server_config
    server_config = config

    # TODO: This is a hack to get the config into the environment variables
    # in order to run multiple worker instances
    os.environ["MOCK_SERVER_TOKENIZER_MODELS"] = json.dumps(config.tokenizer_models)
    os.environ["MOCK_SERVER_TTFT"] = str(config.ttft)
    os.environ["MOCK_SERVER_ITL"] = str(config.itl)
    os.environ["MOCK_SERVER_LOG_LEVEL"] = config.log_level
    os.environ["MOCK_SERVER_HOST"] = config.host
    os.environ["MOCK_SERVER_PORT"] = str(config.port)
    os.environ["MOCK_SERVER_WORKERS"] = str(config.workers)
    os.environ["MOCK_SERVER_ACCESS_LOGS"] = str(config.access_logs)
    os.environ["MOCK_SERVER_FALLBACK_TOKENIZER"] = str(config.fallback_tokenizer)


def extract_user_prompt(messages: list[ChatMessage]) -> str:
    """Extract the user prompt from chat messages."""
    # Combine all user messages for tokenization
    user_messages = [msg.content for msg in messages if msg.role == Role.USER]
    return "\n".join(user_messages) if user_messages else ""


async def generate_streaming_response(
    request: ChatCompletionRequest,
    input_tokens: list,
    request_id: str,
    created_timestamp: int,
    start_time: float,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response."""

    previous_time = start_time
    # Send tokens one by one
    for i, token in enumerate(input_tokens):
        if i == 0:
            # Wait for time to first token with precise timing
            target_time = start_time + (server_config.ttft / 1000.0)
        if i > 0:
            target_time = previous_time + (server_config.itl / 1000.0)

        to_sleep = target_time - perf_counter()
        if to_sleep > 0:
            await asyncio.sleep(to_sleep)

        # Update previous time to calculate next inter-token latency
        previous_time = target_time

        # Create streaming response chunk
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created_timestamp,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={"content": token},
                    finish_reason="stop" if i == len(input_tokens) - 1 else None,
                )
            ],
        )

        yield f"data: {chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/configure")
async def configure(request: ConfigureMessage):
    """Configure the server."""
    if request.itl is not None:
        server_config.itl = request.itl
    if request.ttft is not None:
        server_config.ttft = request.ttft
    if request.tokenizer_models is not None:
        logger.info(f"Loading tokenizer models: {request.tokenizer_models}")
        tokenizer_service.load_tokenizers(request.tokenizer_models)
        logger.info("Tokenizer models loaded successfully")
    if request.fallback_tokenizer is not None:
        tokenizer_service.load_tokenizers([request.fallback_tokenizer])
        tokenizer_service.set_fallback_tokenizer(request.fallback_tokenizer)
        logger.info(f"Fallback tokenizer set to {request.fallback_tokenizer}")
    return {"status": "configured", "config": server_config.model_dump()}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    start_time = perf_counter()
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_timestamp = int(time.time())

    user_prompt = extract_user_prompt(request.messages)
    try:
        # Tokenize the user prompt using the requested model's tokenizer
        tokens = tokenizer_service.tokenize(user_prompt, request.model)
    except Exception as e:
        # If the tokenizer fails, return a 404 error to simulate model not found
        raise HTTPException(
            status_code=404,
            detail="Model Not Found",
        ) from e

    if request.max_completion_tokens is not None:
        tokens = tokens[: request.max_completion_tokens]

    if request.stream:
        # Return streaming response
        return StreamingResponse(
            generate_streaming_response(
                request,
                tokens,
                request_id,
                created_timestamp,
                start_time,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # Return non-streaming response

        # Simulate processing time for all tokens with precise timing
        ttft_time = start_time + (server_config.ttft / 1000.0)
        token_processing_time = (len(tokens) - 1) * server_config.itl / 1000.0

        target_time = ttft_time + token_processing_time

        if target_time > perf_counter():
            await asyncio.sleep(target_time - perf_counter())

        # Reconstruct the response text
        response_text = "".join(tokens)

        # Count tokens for usage statistics
        prompt_tokens = tokenizer_service.count_tokens(user_prompt, request.model)
        completion_tokens = len(tokens)

        response = ChatCompletionResponse(
            id=request_id,
            created=created_timestamp,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role=Role.ASSISTANT, content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

        return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "config": server_config.model_dump()}


@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "message": "AIPerf Integration Test Server",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health": "/health",
            "configure": "/configure",
        },
        "config": server_config.model_dump(),
    }
