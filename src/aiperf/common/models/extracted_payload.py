# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class ExtractedPayload(AIPerfBaseModel):
    """Single-pass extraction result for tokenisation + media accounting.

    Returned by ``BaseEndpoint.extract_payload_inputs``: tokenisable text in
    ``texts`` plus per-modality counts populated as the walk encounters
    ``image_url`` / ``input_audio`` / ``video_url`` parts. One ``orjson.loads``
    plus one O(n) walk yields everything downstream needs.
    """

    texts: list[str] = Field(
        default_factory=list,
        description="Tokenisable text strings (prompt content, instructions, "
        "tool schemas, replayed assistant tool_calls).",
    )
    image_count: int = Field(
        default=0,
        description="Count of image content parts in the payload.",
    )
    audio_count: int = Field(
        default=0,
        description="Count of audio content parts in the payload.",
    )
    video_count: int = Field(
        default=0,
        description="Count of video content parts in the payload.",
    )
    pretokenised_token_count: int = Field(
        default=0,
        description="Token count contributed by pre-tokenised input shapes "
        "(OpenAI embeddings ``input: list[list[int]]`` and ``input: list[int]``). "
        "These bypass the tokeniser entirely - the count is the sum of inner "
        "list lengths and is added to ISL (Input Sequence Length) by the "
        "consumer alongside any ``texts`` it tokenises.",
    )
