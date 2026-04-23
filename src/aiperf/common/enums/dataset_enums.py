# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ConnectionReuseStrategy(CaseInsensitiveStrEnum):
    """Transport connection reuse strategy. Controls how and when connections are reused across requests."""

    POOLED = "pooled"
    """Connections are pooled and reused across all requests"""

    NEVER = "never"
    """New connection for each request, closed after response"""

    STICKY_USER_SESSIONS = "sticky-user-sessions"
    """Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing)"""


class ConversationContextMode(CaseInsensitiveStrEnum):
    """Controls how prior turns are accumulated in multi-turn conversations.

    Two dimensions determine behavior:

    - **Turn format**: ``DELTAS`` (incremental per-turn content) vs
      ``MESSAGE_ARRAY`` (each turn carries its complete message list).
    - **Response inclusion**: ``WITH_RESPONSES`` (pre-canned assistant turns
      are present in the dataset) vs ``WITHOUT_RESPONSES`` (only user content;
      live inference responses are captured at runtime).
    """

    DELTAS_WITHOUT_RESPONSES = "deltas_without_responses"
    """Standard multi-turn chat. Each dataset turn is a user-only delta.
    AIPerf accumulates turns and threads live inference responses into the history."""

    DELTAS_WITH_RESPONSES = "deltas_with_responses"
    """Delta-compressed prompts. Each dataset turn is a delta that may include
    pre-canned assistant responses. AIPerf accumulates but discards live responses."""

    MESSAGE_ARRAY_WITH_RESPONSES = "message_array_with_responses"
    """Self-contained prompts. Each turn carries a complete message array (including
    assistant responses) and is sent as-is. Default for Mooncake traces with
    pre-built ``messages`` arrays."""

    MESSAGE_ARRAY_WITHOUT_RESPONSES = "message_array_without_responses"
    """Reserved. Each turn would carry a complete user-only message array, requiring
    live response merging between turns. Not yet implemented."""


CreditPhase = str
"""Type alias for credit phase names. Phases are arbitrary strings (e.g. 'warmup', 'main', 'cooldown')."""


class DatasetFormat(CaseInsensitiveStrEnum):
    """Defines the format of file-based datasets."""

    SINGLE_TURN = "single_turn"
    """Simple prompt-response pairs."""

    MULTI_TURN = "multi_turn"
    """Conversational data with multiple turns."""

    MOONCAKE_TRACE = "mooncake_trace"
    """Mooncake production trace format."""

    RANDOM_POOL = "random_pool"
    """Treat file as a pool for random sampling."""


class DatasetType(CaseInsensitiveStrEnum):
    """Defines the source type for benchmark datasets."""

    SYNTHETIC = "synthetic"
    """Generate synthetic prompts programmatically."""

    FILE = "file"
    """Load prompts from a local file."""

    PUBLIC = "public"
    """Use a well-known public dataset."""

    COMPOSED = "composed"
    """Combine file-based data with synthetic augmentation."""


class ModelSelectionStrategy(CaseInsensitiveStrEnum):
    """Strategy for selecting the model to use for the request."""

    ROUND_ROBIN = "round_robin"
    """Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models)."""

    RANDOM = "random"
    """Randomly select a model for each prompt using uniform distribution."""

    WEIGHTED = "weighted"
    """Select models based on configured weights. Each model's weight determines its selection probability."""


class OslMode(CaseInsensitiveStrEnum):
    """Defines how output sequence length is handled in composed datasets."""

    FILL = "fill"
    """Only apply OSL if the source record lacks it."""

    OVERRIDE = "override"
    """Always use OSL from augmentation config."""


class PromptSource(CaseInsensitiveStrEnum):
    SYNTHETIC = "synthetic"
    FILE = "file"
    PAYLOAD = "payload"


class SweepType(CaseInsensitiveStrEnum):
    """Defines the sweep strategy for parameter exploration."""

    GRID = "grid"
    """All combinations of variable values (Cartesian product)."""

    SCENARIOS = "scenarios"
    """Hand-picked configurations merged with base."""

    SEQUENTIAL = "sequential"
    """Ordered parameter sets applied one at a time."""
