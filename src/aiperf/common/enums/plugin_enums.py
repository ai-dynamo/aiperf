# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class AIPerfUIType(CaseInsensitiveStrEnum):
    """AIPerf UI types supported by AIPerf.

    Simple string enum for AIPerf UI types. All metadata is retrieved dynamically
    from the registered AIPerf UI classes via AIPerfUIFactory.metadata().
    """

    NONE = "none"
    """No UI"""

    SIMPLE = "simple"
    """Simple UI type using progress bars."""

    DASHBOARD = "dashboard"
    """Complete dashboard UI with real-time metrics and telemetry."""


class EndpointType(CaseInsensitiveStrEnum):
    """Endpoint types supported by AIPerf.

    Simple string enum for endpoint types. All metadata is retrieved dynamically
    from the registered endpoint classes via EndpointFactory.metadata().
    """

    CHAT = "chat"
    """OpenAI Chat Completions API. Supports multi-modal inputs (text, images, audio, video) and streaming."""

    COMPLETIONS = "completions"
    """OpenAI Completions API. Legacy text completion endpoint with streaming support."""

    COHERE_RANKINGS = "cohere_rankings"
    """Cohere Rerank API. Ranks passages by relevance to a query."""

    EMBEDDINGS = "embeddings"
    """OpenAI Embeddings API. Generates vector embeddings for text inputs."""

    HF_TEI_RANKINGS = "hf_tei_rankings"
    """HuggingFace Text Embeddings Inference (TEI) Rankings API. Reranks passages based on query relevance."""

    HUGGINGFACE_GENERATE = "huggingface_generate"
    """HuggingFace Text Generation Inference (TGI) API. Supports both /generate and /generate_stream endpoints."""

    IMAGE_GENERATION = "image_generation"
    """OpenAI Image Generation API. Generates images from text prompts (e.g., FLUX.1)."""

    NIM_EMBEDDINGS = "nim_embeddings"
    """NVIDIA NIM Embeddings API. Generates vector embeddings for text (and image inputs)."""

    NIM_RANKINGS = "nim_rankings"
    """NVIDIA NIM Rankings API. Ranks passages by relevance scores for a given query."""

    SOLIDO_RAG = "solido_rag"
    """SOLIDO RAG API. Retrieval-Augmented Generation endpoint with filter and inference model support."""

    TEMPLATE = "template"
    """Custom template endpoint. Uses Jinja2 templates for flexible payload formatting."""


class TransportType(CaseInsensitiveStrEnum):
    """The various types of transports for an endpoint."""

    HTTP = "http"
    """HTTP/1.1 transport using aiohttp. Supports connection pooling, TCP optimization, and Server-Sent Events (SSE) for streaming."""


class ConnectionReuseStrategy(CaseInsensitiveStrEnum):
    """Transport connection reuse strategy. Controls how and when connections are reused across requests."""

    POOLED = "pooled"
    """Connections are pooled and reused across all requests"""

    NEVER = "never"
    """New connection for each request, closed after response"""

    STICKY_USER_SESSIONS = "sticky-user-sessions"
    """Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing)"""


class URLSelectionStrategy(CaseInsensitiveStrEnum):
    """Strategy for selecting URLs when multiple endpoint URLs are provided.

    Simple string enum for URL selection strategies. Used for load balancing
    across multiple inference server instances.
    """

    ROUND_ROBIN = "round_robin"
    """Distribute requests evenly across URLs in sequential order. Each request goes to the next URL in the list, wrapping around when the end is reached."""
