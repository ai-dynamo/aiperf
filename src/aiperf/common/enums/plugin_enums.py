# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class AIPerfUIType(CaseInsensitiveStrEnum):
    """AIPerf UI types supported by AIPerf.

    Simple string enum for AIPerf UI types. All metadata is retrieved dynamically
    from the registered AIPerf UI classes via AIPerfUIFactory.metadata().
    """

    NONE = "none"
    SIMPLE = "simple"
    DASHBOARD = "dashboard"


class EndpointType(CaseInsensitiveStrEnum):
    """Endpoint types supported by AIPerf.

    Simple string enum for endpoint types. All metadata is retrieved dynamically
    from the registered endpoint classes via EndpointFactory.metadata().
    """

    CHAT = "chat"
    COMPLETIONS = "completions"
    COHERE_RANKINGS = "cohere_rankings"
    EMBEDDINGS = "embeddings"
    HF_TEI_RANKINGS = "hf_tei_rankings"
    HUGGINGFACE_GENERATE = "huggingface_generate"
    IMAGE_GENERATION = "image_generation"
    NIM_RANKINGS = "nim_rankings"
    SOLIDO_RAG = "solido_rag"
    TEMPLATE = "template"


class TransportType(CaseInsensitiveStrEnum):
    """The various types of transports for an endpoint."""

    HTTP = "http"


class ConnectionReuseStrategy(CaseInsensitiveStrEnum):
    """Transport connection reuse strategy.

    Controls how connections are reused across requests:
    - POOLED: Connections are pooled and reused across all requests (default)
    - NEVER: New connection for each request, closed after response
    - STICKY_USER_SESSIONS: Connection persists across turns of a multi-turn
      conversation, closed on final turn (enables sticky load balancing)
    """

    POOLED = "pooled"
    NEVER = "never"
    STICKY_USER_SESSIONS = "sticky-user-sessions"
