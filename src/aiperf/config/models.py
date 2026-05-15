# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Models configuration

This module hosts the model-selection Pydantic configs (per-model override,
advanced model item, weighted-strategy validation). Other top-level config
sections live in sibling submodules to keep any one file under the
ergonomics file-size cap:

* :mod:`aiperf.config.tokenizer`         — tokenizer config
* :mod:`aiperf.config.logging`           — logging config
* :mod:`aiperf.config.slos`              — SLOs type alias
* :mod:`aiperf.config.runtime`           — runtime config
* :mod:`aiperf.config.comm.inputs`       — IPC/TCP/DualBind communication configs
* :mod:`aiperf.config.sweep.multi_run`   — multi-run trial mechanics + convergence
* :mod:`aiperf.config.accuracy`          — accuracy benchmarking config
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.config.base import BaseConfig


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
            "[Currently a no-op: no selection strategy consumes this field. "
            "Accepted for forward-compatibility / declarative documentation.]",
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
        """Validate weights for the weighted selection strategy.

        Enforces two invariants when ``strategy == WEIGHTED``:

        1. Every model item has an explicit ``weight`` (no ``None`` entries).
        2. The sum of weights is within ``[0.99, 1.01]`` (1.0 +/- 0.01).
           Weights are validated, not auto-normalized; out-of-range sums
           raise ``ValueError`` rather than being rescaled.
        """
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


__all__ = [
    "ModelItem",
    "ModelsAdvanced",
    "TokenizerOverride",
]
