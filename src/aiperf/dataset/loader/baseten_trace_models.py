# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema model and session-key helpers for Baseten Parquet traces."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Any

from pydantic import Field, field_validator

from aiperf.common.models import AIPerfBaseModel

METADATA_COLUMNS_TIME = "timestamp_start_unix_ms"
METADATA_COLUMNS_SESSION = "provided_session_id"
METADATA_COLUMNS_POOR_MAN_SESSION = "poor_man_session_id"
METADATA_COLUMNS = {
    METADATA_COLUMNS_TIME,
    METADATA_COLUMNS_SESSION,
    METADATA_COLUMNS_POOR_MAN_SESSION,
}

REQUIRED_COLUMNS = {
    METADATA_COLUMNS_TIME,
    "prompt",
    "input_tokens",
    "output_tokens",
}

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
RequestCanceledInt = Annotated[int, Field(ge=0, le=1)]


class BasetenTrace(AIPerfBaseModel):
    """Schema for Baseten completion traces exported as Parquet."""

    timestamp_start_unix_ms: NonNegativeInt = Field(
        description="Recorded request start timestamp in Unix milliseconds."
    )
    prompt: str = Field(description="Literal completion prompt sent to the server.")
    input_tokens: NonNegativeInt = Field(description="Recorded prompt token count.")
    output_tokens: NonNegativeInt = Field(
        description="Recorded completion token count."
    )
    total_hashes: list[NonNegativeInt] = Field(
        default_factory=list,
        description="Optional KV-cache block hashes aligned to block_size.",
    )
    provided_session_id: str | NonNegativeInt | None = Field(
        default=None,
        description="Session identifier exported directly from the source trace.",
    )
    poor_man_session_id: NonNegativeInt | None = Field(
        default=None,
        description="Fallback derived session identifier.",
    )
    duration_e2e_ms: NonNegativeInt | None = Field(
        default=None,
        description="Recorded end-to-end request duration in milliseconds.",
    )
    duration_ttft_ms: NonNegativeInt | None = Field(
        default=None,
        description="Recorded time to first token in milliseconds.",
    )
    request_canceled: RequestCanceledInt | None = Field(
        default=None,
        description="Whether the source request was canceled.",
    )
    cached_tokens_reference: NonNegativeInt | None = Field(
        default=None,
        description="Recorded reference cached-token count.",
    )
    model_name: str | None = Field(
        default=None,
        description="Model name recorded in the source trace.",
    )
    org_id: str | NonNegativeInt | None = Field(
        default=None,
        description="Organization identifier recorded in the source trace.",
    )
    block_size: PositiveInt | None = Field(
        default=None,
        description="KV-cache block size associated with total_hashes.",
    )
    features: str | None = Field(default=None, description="Opaque feature metadata.")
    speculation_ratio: NonNegativeFloat | None = Field(
        default=None,
        description="Average tokens per decode iteration.",
    )
    output_text: str | None = Field(
        default=None,
        description="Recorded completion text retained for offline validation.",
    )
    dataset_version: str | None = Field(
        default=None,
        alias="__version__",
        description="Source dataset version.",
    )
    total_hashes_len: NonNegativeInt | None = Field(
        default=None,
        description="Recorded total_hashes length, when exported separately.",
    )

    timestamp: NonNegativeInt | NonNegativeFloat | None = Field(
        default=None,
        description="Normalized timestamp in milliseconds since the first event.",
    )
    delay: NonNegativeFloat | None = Field(
        default=None,
        description="Per-turn replay delay in ms, set on continuation turns under "
        "back-pressure: turn N+1 fires this long after turn N completes.",
    )
    input_length: NonNegativeInt | None = Field(
        default=None,
        description="Alias field used by shared trace filtering logic.",
    )
    output_length: NonNegativeInt | None = Field(
        default=None,
        description="Alias field used by shared trace filtering logic.",
    )
    text_input: str | None = Field(
        default=None,
        description="Alias field used by shared trace conversation conversion logic.",
    )
    hash_ids: list[NonNegativeInt] | None = Field(
        default=None,
        description="Alias field used by per-turn request-body forwarding.",
    )
    request_body: dict[str, Any] | None = Field(
        default=None,
        description="Optional per-row payload fields to merge into the outgoing request.",
    )

    @field_validator("total_hashes", mode="before")
    @classmethod
    def _coerce_null_hashes(cls, value: Any) -> Any:
        if value is None:
            return []
        return value


def _score_session_groups(
    session_ids: list[str | int | None],
) -> tuple[int, int]:
    counts = Counter(session_id for session_id in session_ids if session_id is not None)
    repeated_group_sizes = [count for count in counts.values() if count > 1]
    return (sum(repeated_group_sizes), len(repeated_group_sizes))


def choose_baseten_session_key(
    provided_session_ids: list[str | int | None],
    poor_man_session_ids: list[int | None],
) -> str | None:
    """Return the session column with the strongest repeated-session signal."""
    provided_score = _score_session_groups(provided_session_ids)
    poor_score = _score_session_groups(poor_man_session_ids)

    if provided_score > poor_score and provided_score[0] > 0:
        return METADATA_COLUMNS_SESSION
    if poor_score > provided_score and poor_score[0] > 0:
        return METADATA_COLUMNS_POOR_MAN_SESSION
    if provided_score == poor_score and provided_score[0] > 0:
        return METADATA_COLUMNS_SESSION
    return None
