# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core model/tokenizer configuration models and SLOs type alias.

Split out of ``models.py`` so the public module stays under the ergonomics
file-size cap. Re-exported via :mod:`aiperf.config.models`.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.config.base import BaseConfig
from aiperf.config.tokenizer import TokenizerConfig as TokenizerConfig


class TokenizerOverride(BaseConfig):
    """
    Per-model tokenizer override configuration.

    Allows specifying a different tokenizer for a specific model,
    useful when models require specialized tokenization.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str,
        Field(description="HuggingFace tokenizer identifier or local filesystem path."),
    ]


class ModelItem(BaseConfig):
    """
    Configuration for a single model in advanced models configuration.

    Used when the models section uses the advanced format with
    explicit items, weights, and per-model settings.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str,
        Field(
            min_length=1,
            description="Model name or identifier as known to the inference server.",
        ),
    ]

    weight: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=1.0,
            default=None,
            description="Selection weight for weighted strategy (0.0-1.0). "
            "Weights must sum to 1.0 (+/-0.01) across all models; they are "
            "validated, not auto-normalized. "
            "Example: weight=0.7 means ~70%% of requests to this model.",
        ),
    ]

    lora: Annotated[
        str | None,
        Field(
            default=None,
            description="LoRA adapter name to load with this model. "
            "Server must support dynamic LoRA adapter loading.",
        ),
    ]

    modalities: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="List of input modalities this model supports. "
            "Informational today: no selection strategy consumes it yet. "
            "Valid values: 'text', 'image', 'audio', 'video'.",
        ),
    ]

    tokenizer: Annotated[
        TokenizerOverride | None,
        Field(
            default=None,
            description="Per-model tokenizer override. "
            "Use when this model requires a different tokenizer than global config.",
        ),
    ]


class ModelsAdvanced(BaseConfig):
    """
    Advanced models configuration with selection strategy and item details.

    Use this format when you need weighted routing, LoRA adapters,
    or per-model tokenizer overrides.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Annotated[
        ModelSelectionStrategy,
        Field(
            default=ModelSelectionStrategy.ROUND_ROBIN,
            description="Strategy for selecting models when multiple are configured. "
            "round_robin cycles through models, random selects randomly, "
            "weighted uses configured weights.",
        ),
    ]

    items: Annotated[
        list[ModelItem],
        Field(
            min_length=1,
            description="List of model configurations. At least one model required.",
        ),
    ]

    @model_validator(mode="after")
    def validate_weights_for_weighted_strategy(self) -> ModelsAdvanced:
        """Ensure weights are provided when using weighted strategy."""
        if self.strategy == ModelSelectionStrategy.WEIGHTED:
            if not all(item.weight is not None for item in self.items):
                raise ValueError(
                    "All models must have weights specified when using weighted strategy"
                )
            total_weight = sum(
                item.weight for item in self.items if item.weight is not None
            )
            if not (0.99 <= total_weight <= 1.01):
                raise ValueError(f"Model weights must sum to 1.0, got {total_weight}")
        return self


# SLOs is a generic dict allowing any metric name with a threshold value.
# Common metrics: request_latency, time_to_first_token, inter_token_latency, tokens_per_second
SLOsConfig = dict[str, float]
"""
SLOs (Service Level Objectives) configuration as a generic dict.

Maps metric names to threshold values (in milliseconds for latency metrics).
A request is counted as "good" only if it meets ALL specified thresholds.

Example:
    slos:
      request_latency: 500       # max 500ms end-to-end latency
      time_to_first_token: 100   # max 100ms TTFT
      inter_token_latency: 15    # max 15ms between tokens
      tokens_per_second: 50      # min 50 tokens/second
"""
