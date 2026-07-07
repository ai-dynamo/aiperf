# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve whether client-side input tokenization is disabled for a run.

``resolve_disable_tokenization`` is the single source of truth shared with
``InferenceResultParser`` (whose ISL counting reads ``payload_bytes``), so the
parser's ISL path always agrees about whether client-side tokenization runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.plugin.schema.schemas import EndpointMetadata

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig


def resolve_disable_tokenization(
    cfg: BenchmarkConfig, endpoint_meta: EndpointMetadata
) -> bool:
    """Whether client-side input tokenization is disabled for this run.

    Single source of truth shared with ``InferenceResultParser`` (whose ISL
    counting reads ``payload_bytes``): tokenization is off when the user
    requested server-reported counts, or the endpoint neither produces nor
    tokenizes tokens.

    Args:
        cfg: The validated benchmark config for the run.
        endpoint_meta: Plugin metadata for the run's endpoint type.

    Returns:
        True when no client-side tokenization will run.
    """
    return cfg.endpoint.use_server_token_count or (
        not endpoint_meta.produces_tokens and not endpoint_meta.tokenizes_input
    )
