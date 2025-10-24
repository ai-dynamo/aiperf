# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    COHERE_RERANK = "cohere_rerank"
    EMBEDDINGS = "embeddings"
    HF_TEI_RERANKER = "hf_tei_reranker"
    NIM_RANKINGS = "nim_rankings"


class TransportType(CaseInsensitiveStrEnum):
    """The various types of transports for an endpoint."""

    HTTP = "http"
