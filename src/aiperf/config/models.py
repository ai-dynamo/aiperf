# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

This module hosts the core model/tokenizer Pydantic configs and re-exports the
remaining configuration models for the AIPerf YAML configuration system.
Implementations for non-core groups live in sibling submodules to keep any one
file under the ergonomics file-size cap:

* :mod:`aiperf.config.comm.inputs`      — IPC/TCP/DualBind communication configs
* :mod:`aiperf.config.runtime`           — runtime and logging configs
* :mod:`aiperf.config.sweep.multi_run`         — multi-run trial mechanics + convergence
* :mod:`aiperf.config.accuracy`          — accuracy benchmarking config
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.config.accuracy import AccuracyConfig
from aiperf.config.base import BaseConfig
from aiperf.config.comm.inputs import (
    CommunicationConfig,
    DualBindCommunicationConfig,
    IpcCommunicationConfig,
    TcpCommunicationConfig,
    TcpProxyConfig,
)
from aiperf.config.runtime import LoggingConfig, RuntimeConfig
from aiperf.config.sweep.multi_run import ConvergenceConfig, MultiRunConfig


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


class TokenizerConfig(BaseConfig):
    """
    Tokenizer configuration for token counting and prompt generation.

    AIPerf uses a HuggingFace tokenizer for accurate token counting,
    which is essential for ISL/OSL enforcement and metrics calculation.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str | None,
        Field(
            default=None,
            description="HuggingFace tokenizer identifier, local filesystem path, or `builtin` "
            "for a zero-network-access tokenizer backed by tiktoken (o200k_base encoding). "
            "Should match the model's tokenizer for accurate token counts. "
            "If `--tokenizer` is not set and the model name looks like an obvious placeholder "
            "(e.g. `mock-model`, `test-model`, `fake-model`), AIPerf substitutes `builtin` automatically "
            "and emits a warning. "
            "Example: 'meta-llama/Llama-3.1-8B-Instruct'",
        ),
    ]

    revision: Annotated[
        str,
        Field(
            default="main",
            description="Model revision to use: branch name, tag, or commit hash. "
            "Use for version pinning to ensure reproducibility.",
        ),
    ]

    trust_remote_code: Annotated[
        bool,
        Field(
            default=False,
            description="Allow execution of custom tokenizer code from the repository. "
            "Required for some models but poses security risk. "
            "Only enable for trusted sources.",
        ),
    ]

    resolved_names: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            exclude=True,
            description="Pre-resolved tokenizer names from alias resolution. "
            "[runtime-only; populated by the CLI or WorkerGroupManager after "
            "tokenizer validation. Excluded from JSON/YAML serialization. Do not "
            "set in a CR spec — any user value is ignored.]",
        ),
    ]

    def get_tokenizer_name_for_model(self, model_name: str) -> str:
        """Get the tokenizer name to use for a given model.

        Resolution order:
        1. Pre-resolved name from `resolved_names` (set by CLI after alias resolution)
        2. Explicitly configured tokenizer name
        3. The model name itself (assumes model repo contains tokenizer)
        """
        if self.resolved_names and model_name in self.resolved_names:
            return self.resolved_names[model_name]
        return self.name or model_name

    @property
    def should_resolve_alias(self) -> bool:
        """Whether alias resolution should be performed when loading tokenizers.

        Returns False if `resolved_names` is set (CLI already resolved aliases),
        True otherwise to enable HuggingFace Hub alias resolution.
        """
        return self.resolved_names is None


__all__ = [
    # Accuracy benchmarking
    "AccuracyConfig",
    # Communication
    "CommunicationConfig",
    # Convergence (nested under MultiRunConfig)
    "ConvergenceConfig",
    "DualBindCommunicationConfig",
    "IpcCommunicationConfig",
    # Logging
    "LoggingConfig",
    # Models
    "ModelItem",
    "ModelsAdvanced",
    # Multi-run
    "MultiRunConfig",
    # Runtime
    "RuntimeConfig",
    # SLOs
    "SLOsConfig",
    "TcpCommunicationConfig",
    "TcpProxyConfig",
    # Tokenizer
    "TokenizerConfig",
    "TokenizerOverride",
]
