# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.endpoints.base_endpoint import (
    BaseEndpoint,
)
from aiperf.endpoints.nim_rankings import (
    RankingsEndpoint,
)
from aiperf.endpoints.openai_chat import (
    ChatEndpoint,
)
from aiperf.endpoints.openai_completions import (
    CompletionsEndpoint,
)
from aiperf.endpoints.openai_embeddings import (
    EmbeddingsEndpoint,
)

__all__ = [
    "BaseEndpoint",
    "ChatEndpoint",
    "CompletionsEndpoint",
    "EmbeddingsEndpoint",
    "RankingsEndpoint",
]
