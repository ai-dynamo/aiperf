# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for Agentic Code session dataset generation."""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.models import AIPerfBaseModel


class PercentileStats(AIPerfBaseModel):
    """Descriptive statistics with percentile breakdown."""

    count: int = Field(description="Number of observations")
    mean: float = Field(description="Arithmetic mean")
    std: float = Field(description="Standard deviation")
    median: float = Field(description="50th percentile")
    p05: float = Field(description="5th percentile")
    p25: float = Field(description="25th percentile")
    p75: float = Field(description="75th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")


def _maybe_round(value: float, digits: int | None) -> float:
    return value if digits is None else round(value, digits)


def percentile_stats(arr: np.ndarray, digits: int | None = 2) -> PercentileStats:
    """Compute PercentileStats from a numpy array."""
    return PercentileStats(
        count=len(arr),
        mean=_maybe_round(float(np.mean(arr)), digits),
        std=_maybe_round(float(np.std(arr)), digits),
        median=_maybe_round(float(np.median(arr)), digits),
        p05=_maybe_round(float(np.percentile(arr, 5)), digits),
        p25=_maybe_round(float(np.percentile(arr, 25)), digits),
        p75=_maybe_round(float(np.percentile(arr, 75)), digits),
        p95=_maybe_round(float(np.percentile(arr, 95)), digits),
        p99=_maybe_round(float(np.percentile(arr, 99)), digits),
    )


class SessionEndReason(str, Enum):
    """Why a session ended."""

    FORCED_RETIRE = "forced_retire"
    PROBABILISTIC_RESET = "probabilistic_reset"
    RESTART_SPLIT = "restart_split"
    TARGET_TURN_COUNT = "target_turn_count"


class LognormalParams(AIPerfBaseModel):
    """Lognormal distribution parameters with real-space summary statistics.

    Can be constructed in two ways:
    1. Full: mu, sigma, mean, median all provided (e.g. from manifest.json or fit-stats)
    2. Simplified: just mean and median — mu/sigma auto-computed via model validator
    """

    mu: float | None = Field(default=None, description="Log-space mean")
    sigma: float | None = Field(
        default=None, ge=0.0, description="Log-space standard deviation"
    )
    mean: float = Field(gt=0.0, description="Real-space mean (derived)")
    median: float = Field(gt=0.0, description="Real-space median (derived)")
    min: float | None = Field(
        default=None, gt=0.0, description="Hard lower bound (rejection sampled)"
    )
    max: float | None = Field(
        default=None, gt=0.0, description="Hard upper bound (rejection sampled)"
    )

    @model_validator(mode="after")
    def compute_mu_sigma(self) -> LognormalParams:
        if self.mean < self.median:
            raise ValueError(
                f"mean ({self.mean}) must be >= median ({self.median}) for lognormal"
            )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        if (self.mu is None) != (self.sigma is None):
            raise ValueError("mu and sigma must be supplied as a pair")
        if self.mu is not None and not math.isfinite(self.mu):
            raise ValueError("mu must be finite")
        if self.sigma is not None and not math.isfinite(self.sigma):
            raise ValueError("sigma must be finite")
        if self.mu is None:
            self.mu = math.log(self.median)
            ratio = self.mean / self.median
            self.sigma = math.sqrt(2.0 * math.log(ratio)) if ratio > 1.0 else 0.0
        return self


class NewTokensPerTurnConfig(LognormalParams):
    """Lognormal config for new tokens per turn with truncation-bias correction."""

    bias: float = Field(
        default=1.0,
        gt=0.0,
        description="Multiplier on the target mean/median to compensate for truncation bias",
    )


def _default_agentic_delay() -> LognormalParams:
    return LognormalParams(mean=2_500, median=1_800)


def _default_human_delay() -> LognormalParams:
    return LognormalParams(mean=40_000, median=25_000)


class MixtureDelayConfig(AIPerfBaseModel):
    """Two-component mixture model for inter-turn delays.

    Agentic turns (tool-call follow-ups) are fast; human turns are slow.
    A Bernoulli draw selects which component to sample from.
    """

    agentic_fraction: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Probability of sampling the fast agentic delay",
    )
    agentic_delay: LognormalParams = Field(
        default_factory=_default_agentic_delay,
        description="Fast delay distribution (tool-call follow-ups)",
    )
    human_delay: LognormalParams = Field(
        default_factory=_default_human_delay,
        description="Slow delay distribution (human think time)",
    )
    max: float | None = Field(
        default=None, ge=0.0, description="Hard upper clip on sampled delay (ms)"
    )


class ResetConfig(AIPerfBaseModel):
    """Context-dependent reset probability.

    Models explicit continuation, repo-context edits, and TTL expiry.
    P(reset) = base_probability * (1 + (context_scaling - 1) * input_length / max_prompt_tokens)
    """

    base_probability: float = Field(
        default=0.02, ge=0.0, le=1.0, description="Base reset chance per turn"
    )
    context_scaling: float = Field(
        default=2.0, ge=1.0, description="Multiplier at max_prompt_tokens"
    )


class TurnCountConfig(AIPerfBaseModel):
    """Explicit target distribution for turns per session."""

    mean: int = Field(gt=0, description="Target mean turns per session")
    median: int = Field(gt=0, description="Target median turns per session")
    min: int = Field(gt=0, description="Hard lower bound on sampled turn count")
    max: int = Field(gt=0, description="Hard upper bound on sampled turn count")
    allow_truncation: bool = Field(
        default=False,
        description="Allow sessions to end early at the context limit instead of resampling",
    )
    max_session_attempts: int | None = Field(
        default=None,
        ge=1,
        description="Max full-session retries before failing in exact-turn mode",
    )

    @model_validator(mode="after")
    def validate_ordering(self) -> TurnCountConfig:
        if self.mean < self.median:
            raise ValueError(
                f"mean ({self.mean}) must be >= median ({self.median}) for lognormal"
            )
        if self.min > self.median:
            raise ValueError(f"min ({self.min}) must be <= median ({self.median})")
        if self.mean > self.max:
            raise ValueError(f"mean ({self.mean}) must be <= max ({self.max})")
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        if self.allow_truncation:
            if self.max_session_attempts is not None:
                raise ValueError(
                    "max_session_attempts cannot be set when allow_truncation is true"
                )
        elif self.max_session_attempts is None:
            self.max_session_attempts = 100
        return self

    def to_lognormal(self) -> LognormalParams:
        """Create bounded lognormal params for integer turn sampling."""
        return LognormalParams(
            mean=float(self.mean),
            median=float(self.median),
            min=float(self.min),
            max=float(self.max),
        )


class Layer15GroupConfig(AIPerfBaseModel):
    """Group assignment for L1.5 cache sharing via Zipf distribution."""

    num_groups: int = Field(
        default=50, ge=1, description="Number of distinct groups (repos/projects)"
    )
    zipf_alpha: float = Field(
        default=1.2, ge=0.0, description="Zipf skew parameter (higher = more skewed)"
    )


class CacheLayerConfig(AIPerfBaseModel):
    """Token sizes for the KV cache prefix model.

    L1: Global (tools + system prompt), shared by all sessions.
    L1.5: Group-shared (repo instructions and context), shared within a group.
    L2: Session-specific prefix (initial files), sampled per session.
    L3: Conversation history, grows turn-by-turn (not configured here).
    """

    layer1_tokens: int = Field(
        default=32_000,
        ge=0,
        description="L1: tools + system prompt tokens (globally cached)",
    )
    layer1_5_tokens: int = Field(
        default=20_000,
        ge=0,
        description="L1.5: group-shared prefix tokens (repo instructions and context)",
    )
    layer2: LognormalParams = Field(
        default_factory=lambda: LognormalParams(mean=10_000, median=5_000),
        description="L2: session-specific prefix token distribution",
    )
    layer1_5_groups: Layer15GroupConfig = Field(
        default_factory=lambda: Layer15GroupConfig(),
        description="L1.5 group assignment for prefix sharing",
    )


def _default_new_tokens_per_turn() -> NewTokensPerTurnConfig:
    return NewTokensPerTurnConfig(mean=3_500, median=1_800)


def _default_generation_length() -> LognormalParams:
    return LognormalParams(mean=500, median=300)


class SessionDistributionConfig(BaseConfig):
    """Full configuration for synthesizing Agentic Code sessions.

    initial_context is derived: L1 + L1.5 + sampled L2. Not directly configured.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    new_tokens_per_turn: NewTokensPerTurnConfig = Field(
        default_factory=_default_new_tokens_per_turn,
        description="New tokens added per turn",
    )
    generation_length: LognormalParams = Field(
        default_factory=_default_generation_length,
        description="Output token distribution",
    )
    inter_turn_delay: MixtureDelayConfig = Field(
        default_factory=MixtureDelayConfig, description="Inter-turn delay mixture model"
    )
    reset: ResetConfig | None = Field(
        default_factory=ResetConfig, description="Reset probability config"
    )
    turns: TurnCountConfig | None = Field(
        default=None,
        description="Explicit turns-per-session mode (mutually exclusive with reset)",
    )
    max_prompt_tokens: int = Field(
        default=200_000, ge=1, description="Context window limit"
    )
    block_size: int = Field(
        default=512, ge=1, description="KV cache page size in tokens"
    )
    cache: CacheLayerConfig = Field(
        default_factory=CacheLayerConfig, description="Cache layer config"
    )
    restart_initial_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Initial probability that a primary session is split into a restart "
            "continuation. The probability decays linearly to zero across the first "
            "75% of generated primary sessions, so the observed continuation count "
            "is lower than this value."
        ),
    )
    restart_turn_range: list[int] = Field(
        default=[5, 15],
        min_length=2,
        max_length=2,
        description="[min, max) turn index range for restart split point",
    )

    @model_validator(mode="before")
    @classmethod
    def drop_deprecated_system_prompt_tokens(cls, data: object) -> object:
        """Ignore the removed user-facing system_prompt_tokens config key."""
        if isinstance(data, dict):
            data = dict(data)
            data.pop("system_prompt_tokens", None)
            if "restart_fraction" in data:
                restart_fraction = data.pop("restart_fraction")
                restart_probability = data.get("restart_initial_probability")
                if (
                    restart_probability is not None
                    and restart_probability != restart_fraction
                ):
                    raise ValueError(
                        "restart_fraction cannot differ from "
                        "restart_initial_probability"
                    )
                data["restart_initial_probability"] = restart_fraction
            if data.get("turns") is not None and "reset" not in data:
                data["reset"] = None
            ntp = data.get("new_tokens_per_turn")
            if isinstance(ntp, LognormalParams):
                data["new_tokens_per_turn"] = ntp.model_dump()
        return data

    @model_validator(mode="after")
    def validate_turn_mode(self) -> SessionDistributionConfig:
        lo, hi = self.restart_turn_range
        if lo < 1:
            raise ValueError("restart_turn_range minimum must be >= 1")
        if hi <= lo:
            raise ValueError("restart_turn_range must be ordered as [min, max)")

        if self.turns is None:
            return self

        if self.reset is not None:
            raise ValueError("turns mode cannot be combined with reset")

        if self.restart_initial_probability != 0.0:
            raise ValueError(
                "turns mode cannot be combined with restart_initial_probability"
            )

        self._validate_turn_feasibility()
        return self

    def _validate_turn_feasibility(self) -> None:
        """Reject clearly impossible explicit-turn configs up front."""
        if self.turns is None:
            return

        min_l2 = int(self.cache.layer2.min) if self.cache.layer2.min is not None else 1
        min_new_tokens = (
            int(self.new_tokens_per_turn.min)
            if self.new_tokens_per_turn.min is not None
            else 1
        )
        gen_max = self.generation_length.max
        min_output = min(30, int(gen_max)) if gen_max is not None else 30

        input_length = self.cache.layer1_tokens + self.cache.layer1_5_tokens + min_l2
        if input_length >= self.max_prompt_tokens:
            raise ValueError(
                "turns mode is impossible: minimum initial context exceeds max_prompt_tokens"
            )

        if self.turns.allow_truncation:
            return

        for _ in range(1, self.turns.min):
            input_length += min_output + min_new_tokens
            if input_length >= self.max_prompt_tokens:
                raise ValueError(
                    "turns mode is impossible: minimum turn count cannot fit under "
                    "max_prompt_tokens"
                )


class SynthesizedTurn(AIPerfBaseModel):
    """A single synthesized turn within a session."""

    turn_index: int = Field(ge=0, description="Turn number within session")
    input_length: int = Field(ge=1, description="Total input tokens for this turn")
    output_length: int = Field(ge=1, description="Output tokens generated")
    new_tokens: int = Field(ge=0, description="New tokens added since previous turn")
    delay_ms: float = Field(
        ge=0.0, description="Delay before this turn in milliseconds"
    )
    timestamp_ms: float = Field(
        ge=0.0, description="Absolute timestamp in milliseconds"
    )
    hash_ids: list[int] = Field(
        description="KV cache block hash IDs for prefix matching"
    )


class SynthesizedSession(AIPerfBaseModel):
    """A complete synthesized multi-turn session."""

    session_id: str = Field(description="Unique session identifier")
    group_id: int = Field(description="Group index for L1.5 cache sharing")
    turns: list[SynthesizedTurn] = Field(description="Ordered list of turns")
    end_reason: SessionEndReason = Field(description="Why the session ended")
    is_restart_continuation: bool = Field(
        default=False,
        description="True for Session B's created from restart splits",
    )

    @model_validator(mode="after")
    def validate_turns_ordered(self) -> SynthesizedSession:
        if not self.turns:
            raise ValueError("sessions must contain at least one turn")
        for i, turn in enumerate(self.turns):
            if turn.turn_index != i:
                raise ValueError(
                    f"Turn {i} has turn_index={turn.turn_index}, expected {i}"
                )
        return self


class DatasetManifest(AIPerfBaseModel):
    """Metadata written alongside the JSONL dataset."""

    seed: int = Field(description="Random seed used for generation")
    num_sessions: int = Field(ge=1, description="Number of sessions generated")
    config_name: str | None = Field(
        default=None, description="Config name or path used for generation"
    )
    generation_params: SessionDistributionConfig = Field(
        description="Full generation config"
    )


class QualityMetric(AIPerfBaseModel):
    """Observed vs target comparison for one metric with full percentile breakdown."""

    target_mean: float | None = Field(
        default=None, description="Target mean from config"
    )
    target_median: float | None = Field(
        default=None, description="Target median from config"
    )
    observed: PercentileStats = Field(
        description="Full observed distribution statistics"
    )
    pct_error_mean: float | None = Field(
        default=None, description="Absolute percentage error on mean"
    )
    pct_error_median: float | None = Field(
        default=None, description="Absolute percentage error on median"
    )


class SessionEndStats(AIPerfBaseModel):
    """Statistics about how sessions ended."""

    total_sessions: int = Field(description="Total number of sessions")
    forced_retires: int = Field(description="Sessions ended by hitting context limit")
    probabilistic_resets: int = Field(
        description="Sessions ended by probabilistic reset"
    )
    target_turn_completions: int = Field(
        default=0, description="Sessions ended by reaching the explicit turn target"
    )
    restart_splits: int = Field(
        default=0, description="Sessions ended by restart split"
    )
    retire_fraction: float = Field(description="Fraction of forced retires")
    reset_fraction: float = Field(description="Fraction of probabilistic resets")
    target_turn_fraction: float = Field(
        default=0.0,
        description="Fraction of sessions ended by reaching the explicit turn target",
    )
    restart_split_fraction: float = Field(
        default=0.0, description="Fraction of restart splits"
    )
    final_context_utilization: PercentileStats = Field(
        description="Distribution of last-turn input_length / max_prompt_tokens"
    )


class QualityReport(AIPerfBaseModel):
    """Quality report for a generated dataset."""

    config_summary: dict[str, float | int] = Field(
        description="Flat readable config parameters"
    )
    observed_vs_target: dict[str, QualityMetric] = Field(
        description="Per-metric quality checks with percentile breakdowns"
    )
    session_stats: PercentileStats = Field(description="Turns-per-session distribution")
    session_end_stats: SessionEndStats = Field(description="How sessions ended")
