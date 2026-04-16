# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric extraction for Agentic Code dataset reports."""

from __future__ import annotations

from collections import Counter

import numpy as np
from pydantic import Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.agentic_code_gen.models import (
    DatasetManifest,
    PercentileStats,
    QualityMetric,
    QualityReport,
    SessionDistributionConfig,
    SessionEndReason,
    SessionEndStats,
    SynthesizedSession,
    percentile_stats,
)
from aiperf.dataset.agentic_code_gen.reporting.trace import ParsedTurn


def _pct_error(target: float, observed: float) -> float:
    if target == 0:
        return 0.0
    return abs(observed - target) / target * 100.0


def _percentile_stats(arr: np.ndarray) -> PercentileStats:
    """Compute unrounded percentile stats for report displays."""
    return percentile_stats(arr, digits=None)


class TargetComparison(AIPerfBaseModel):
    """Observed stats vs a target mean/median for one metric."""

    metric_name: str = Field(description="Name of the compared metric")
    target_mean: float | None = Field(
        default=None, description="Expected mean from config"
    )
    target_median: float | None = Field(
        default=None, description="Expected median from config"
    )
    observed: PercentileStats = Field(description="Observed descriptive statistics")
    pct_error_mean: float | None = Field(
        default=None, description="Percentage error between observed and target mean"
    )


class ReportData(AIPerfBaseModel):
    """Full report payload."""

    session_count: int = Field(description="Number of sessions in the dataset")
    total_turns: int = Field(description="Total number of turns across all sessions")
    comparisons: list[TargetComparison] = Field(
        description="Target-vs-observed metric comparisons"
    )
    hash_id_block_stats: PercentileStats = Field(
        description="Hash ID block count statistics"
    )
    request_latency_stats: PercentileStats = Field(
        description="Per-turn request latency statistics"
    )
    session_duration_min_stats: PercentileStats = Field(
        description="Session duration in minutes statistics"
    )
    prefix_length_stats: PercentileStats | None = Field(
        default=None, description="Prefix length statistics"
    )
    unique_prompt_length_stats: PercentileStats | None = Field(
        default=None, description="Unique prompt length statistics"
    )
    prefix_ratio_stats: PercentileStats | None = Field(
        default=None, description="Prefix ratio statistics"
    )
    sequential_cache_hit_rate_stats: PercentileStats | None = Field(
        default=None, description="Sequential cache hit rate statistics"
    )
    per_session_cache_hit_rate_stats: PercentileStats | None = Field(
        default=None, description="Per-session cache hit rate statistics"
    )


def extract_metrics(
    sessions: dict[str, list[ParsedTurn]],
    prefill_tps: float = 20_000,
    decode_tps: float = 60,
) -> dict[str, np.ndarray]:
    initial_context: list[float] = []
    new_tokens_per_turn: list[float] = []
    generation_length: list[float] = []
    inter_turn_delay: list[float] = []
    turns_per_session: list[float] = []
    total_isl: list[float] = []
    total_osl: list[float] = []
    hash_id_block_count: list[float] = []
    request_latency_ms: list[float] = []
    session_duration_min: list[float] = []

    for turns in sessions.values():
        turns_per_session.append(float(len(turns)))
        session_lat = 0.0
        for i, turn in enumerate(turns):
            total_isl.append(float(turn.input_length))
            total_osl.append(float(turn.output_length))
            generation_length.append(float(turn.output_length))
            hash_id_block_count.append(float(len(turn.hash_ids)))

            lat = (
                turn.input_length / prefill_tps + turn.output_length / decode_tps
            ) * 1000
            request_latency_ms.append(lat)
            session_lat += turn.delay_ms + lat

            if i == 0:
                initial_context.append(float(turn.input_length))
            else:
                new_tokens_per_turn.append(float(turn.input_length))
                inter_turn_delay.append(turn.delay_ms / 1000.0)

        session_duration_min.append(session_lat / 1000.0 / 60.0)

    return {
        "initial_context": np.array(initial_context),
        "new_tokens_per_turn": np.array(new_tokens_per_turn),
        "generation_length": np.array(generation_length),
        "inter_turn_delay_s": np.array(inter_turn_delay),
        "turns_per_session": np.array(turns_per_session),
        "total_isl": np.array(total_isl),
        "total_osl": np.array(total_osl),
        "hash_id_block_count": np.array(hash_id_block_count),
        "request_latency_ms": np.array(request_latency_ms),
        "request_latency_s": np.array(request_latency_ms) / 1000.0,
        "session_duration_min": np.array(session_duration_min),
    }


def extract_cache_metrics(
    sessions: dict[str, list[ParsedTurn]],
    block_size: int = 512,
) -> dict[str, np.ndarray]:
    """Compute prefix/cache-reuse statistics from hash_ids."""
    all_turns: list[ParsedTurn] = []
    session_boundaries: list[int] = []
    for turns in sessions.values():
        session_boundaries.append(len(all_turns))
        all_turns.extend(turns)
    session_boundary_set = set(session_boundaries)

    hash_counter: Counter[tuple[int, int]] = Counter()
    for turn in all_turns:
        for pos, hid in enumerate(turn.hash_ids):
            hash_counter[(pos, hid)] += 1
    repeated = {k for k, v in hash_counter.items() if v > 1}

    prefix_length: list[float] = []
    unique_prompt_length: list[float] = []
    prefix_ratio: list[float] = []
    sequential_cache_hit_rate: list[float] = []
    per_session_cache_hit_rate: list[float] = []
    global_seen: set[int] = set()
    session_seen: set[int] = set()

    for idx, turn in enumerate(all_turns):
        hash_ids = turn.hash_ids
        input_length = turn.input_length

        repeated_count = sum(
            1 for pos, hid in enumerate(hash_ids) if (pos, hid) in repeated
        )
        prefix_tokens = (
            input_length
            if hash_ids and repeated_count == len(hash_ids)
            else min(repeated_count * block_size, input_length)
        )

        prefix_length.append(float(prefix_tokens))
        unique_prompt_tokens = max(input_length - prefix_tokens, 0)
        unique_prompt_length.append(float(unique_prompt_tokens))
        prefix_ratio.append(prefix_tokens / input_length if input_length > 0 else 0.0)

        sequential_cache_hit_rate.append(_cache_hit_rate(hash_ids, global_seen))
        global_seen.update(hash_ids)

        if idx in session_boundary_set:
            session_seen = set()
        per_session_cache_hit_rate.append(_cache_hit_rate(hash_ids, session_seen))
        session_seen.update(hash_ids)

    return {
        "prefix_length": np.array(prefix_length),
        "unique_prompt_length": np.array(unique_prompt_length),
        "prefix_ratio": np.array(prefix_ratio),
        "sequential_cache_hit_rate": np.array(sequential_cache_hit_rate),
        "per_session_cache_hit_rate": np.array(per_session_cache_hit_rate),
    }


def _cache_hit_rate(hash_ids: list[int], seen: set[int]) -> float:
    """Return prefix cache hit rate for hash_ids against a seen-block set."""
    if not hash_ids:
        return 0.0
    first_unseen = len(hash_ids)
    for idx, hid in enumerate(hash_ids):
        if hid not in seen:
            first_unseen = idx
            break
    return first_unseen / len(hash_ids)


def _target_table(
    manifest: DatasetManifest | None,
) -> list[tuple[str, float | None, float | None, str]]:
    config = (
        manifest.generation_params
        if manifest is not None
        else SessionDistributionConfig()
    )
    cache = config.cache
    turn_target_mean = float(config.turns.mean) if config.turns is not None else None
    turn_target_median = (
        float(config.turns.median) if config.turns is not None else None
    )
    return [
        (
            "initial_context",
            cache.layer1_tokens + cache.layer1_5_tokens + cache.layer2.mean,
            cache.layer1_tokens + cache.layer1_5_tokens + cache.layer2.median,
            "Initial Context (tokens)",
        ),
        (
            "new_tokens_per_turn",
            config.new_tokens_per_turn.mean,
            config.new_tokens_per_turn.median,
            "New Tokens/Turn",
        ),
        (
            "generation_length",
            config.generation_length.mean,
            config.generation_length.median,
            "Generation Length (tokens)",
        ),
        ("inter_turn_delay_s", None, None, "Inter-Turn Delay (s)"),
        ("turns_per_session", turn_target_mean, turn_target_median, "Turns/Session"),
    ]


def build_report_data(
    metrics: dict[str, np.ndarray],
    manifest: DatasetManifest | None = None,
) -> ReportData:
    comparisons: list[TargetComparison] = []
    for key, target_mean, target_median, display in _target_table(manifest):
        arr = metrics.get(key)
        if arr is None or len(arr) == 0:
            continue
        observed = _percentile_stats(arr)
        pct_err = (
            _pct_error(target_mean, observed.mean) if target_mean is not None else None
        )
        comparisons.append(
            TargetComparison(
                metric_name=display,
                target_mean=target_mean,
                target_median=target_median,
                observed=observed,
                pct_error_mean=round(pct_err, 2) if pct_err is not None else None,
            )
        )

    cache_fields: dict[str, PercentileStats | None] = {}
    for field_name, metric_key in [
        ("prefix_length_stats", "prefix_length"),
        ("unique_prompt_length_stats", "unique_prompt_length"),
        ("prefix_ratio_stats", "prefix_ratio"),
        ("sequential_cache_hit_rate_stats", "sequential_cache_hit_rate"),
        ("per_session_cache_hit_rate_stats", "per_session_cache_hit_rate"),
    ]:
        arr = metrics.get(metric_key)
        cache_fields[field_name] = (
            _percentile_stats(arr) if arr is not None and len(arr) > 0 else None
        )

    return ReportData(
        session_count=len(metrics.get("turns_per_session", np.array([]))),
        total_turns=int(metrics.get("total_isl", np.array([])).shape[0]),
        comparisons=comparisons,
        hash_id_block_stats=_percentile_stats(metrics["hash_id_block_count"]),
        request_latency_stats=_percentile_stats(metrics["request_latency_ms"]),
        session_duration_min_stats=_percentile_stats(metrics["session_duration_min"]),
        **cache_fields,
    )


def config_summary(config: SessionDistributionConfig) -> dict[str, float | int]:
    """Flatten config into a readable dict of key parameters."""
    l1 = config.cache.layer1_tokens
    l15 = config.cache.layer1_5_tokens
    summary: dict[str, float | int] = {
        "initial_context_mean": l1 + l15 + config.cache.layer2.mean,
        "initial_context_median": l1 + l15 + config.cache.layer2.median,
        "layer1_tokens": l1,
        "layer1_5_tokens": l15,
        "layer2_mean": config.cache.layer2.mean,
        "layer2_median": config.cache.layer2.median,
        "num_groups": config.cache.layer1_5_groups.num_groups,
        "zipf_alpha": config.cache.layer1_5_groups.zipf_alpha,
        "new_tokens_per_turn_mean": config.new_tokens_per_turn.mean,
        "new_tokens_per_turn_median": config.new_tokens_per_turn.median,
        "generation_length_mean": config.generation_length.mean,
        "generation_length_median": config.generation_length.median,
        "max_prompt_tokens": config.max_prompt_tokens,
        "block_size": config.block_size,
        "inter_turn_delay_agentic_fraction": config.inter_turn_delay.agentic_fraction,
        "inter_turn_delay_agentic_mean_ms": config.inter_turn_delay.agentic_delay.mean,
        "inter_turn_delay_human_mean_ms": config.inter_turn_delay.human_delay.mean,
        "turn_mode_enabled": int(config.turns is not None),
    }
    if config.reset is not None:
        summary["reset_base_probability"] = config.reset.base_probability
        summary["reset_context_scaling"] = config.reset.context_scaling
    if config.turns is not None:
        summary["turns_mean"] = config.turns.mean
        summary["turns_median"] = config.turns.median
        summary["turns_min"] = config.turns.min
        summary["turns_max"] = config.turns.max
        summary["turns_allow_truncation"] = int(config.turns.allow_truncation)
        if config.turns.max_session_attempts is not None:
            summary["turns_max_session_attempts"] = config.turns.max_session_attempts
    return summary


def build_quality_metric(
    arr: np.ndarray,
    target_mean: float | None = None,
    target_median: float | None = None,
) -> QualityMetric:
    """Build a QualityMetric with percentile stats and error calculations."""
    observed = percentile_stats(arr)
    pct_error_mean = (
        round(_pct_error(target_mean, observed.mean), 2)
        if target_mean is not None
        else None
    )
    pct_error_median = (
        round(_pct_error(target_median, observed.median), 2)
        if target_median is not None
        else None
    )
    return QualityMetric(
        target_mean=target_mean,
        target_median=target_median,
        observed=observed,
        pct_error_mean=pct_error_mean,
        pct_error_median=pct_error_median,
    )


def compute_quality_report(
    sessions: list[SynthesizedSession],
    config: SessionDistributionConfig,
) -> QualityReport:
    """Compute quality metrics comparing observed distributions to targets."""
    all_initial_ctx: list[float] = []
    all_new_tokens: list[float] = []
    all_output_lens: list[float] = []
    all_delays: list[float] = []
    turns_per_session: list[float] = []
    final_context_utils: list[float] = []
    forced_retires = 0
    probabilistic_resets = 0
    target_turn_completions = 0
    restart_splits = 0

    for session in sessions:
        turns_per_session.append(float(len(session.turns)))

        if session.end_reason == SessionEndReason.FORCED_RETIRE:
            forced_retires += 1
        elif session.end_reason == SessionEndReason.PROBABILISTIC_RESET:
            probabilistic_resets += 1
        elif session.end_reason == SessionEndReason.TARGET_TURN_COUNT:
            target_turn_completions += 1
        elif session.end_reason == SessionEndReason.RESTART_SPLIT:
            restart_splits += 1

        last_turn = session.turns[-1]
        final_context_utils.append(last_turn.input_length / config.max_prompt_tokens)

        for turn in session.turns:
            if turn.turn_index == 0:
                all_initial_ctx.append(float(turn.input_length))
            else:
                all_new_tokens.append(float(turn.new_tokens))
                all_delays.append(turn.delay_ms)
            all_output_lens.append(float(turn.output_length))

    observed_vs_target: dict[str, QualityMetric] = {}

    if all_initial_ctx:
        l1 = config.cache.layer1_tokens
        l15 = config.cache.layer1_5_tokens
        observed_vs_target["initial_context"] = build_quality_metric(
            np.array(all_initial_ctx),
            target_mean=l1 + l15 + config.cache.layer2.mean,
            target_median=l1 + l15 + config.cache.layer2.median,
        )

    if all_output_lens:
        observed_vs_target["generation_length"] = build_quality_metric(
            np.array(all_output_lens),
            target_mean=config.generation_length.mean,
            target_median=config.generation_length.median,
        )

    if all_new_tokens:
        observed_vs_target["new_tokens_per_turn"] = build_quality_metric(
            np.array(all_new_tokens),
            target_mean=config.new_tokens_per_turn.mean,
            target_median=config.new_tokens_per_turn.median,
        )

    if all_delays:
        observed_vs_target["inter_turn_delay_ms"] = build_quality_metric(
            np.array(all_delays),
            target_mean=None,
            target_median=None,
        )

    if turns_per_session:
        observed_vs_target["turns_per_session"] = build_quality_metric(
            np.array(turns_per_session),
            target_mean=float(config.turns.mean) if config.turns is not None else None,
            target_median=float(config.turns.median)
            if config.turns is not None
            else None,
        )

    tps_arr = np.array(turns_per_session) if turns_per_session else np.array([0.0])
    session_stats = percentile_stats(tps_arr)

    total = len(sessions)
    fcu_arr = np.array(final_context_utils) if final_context_utils else np.array([0.0])
    session_end_stats = SessionEndStats(
        total_sessions=total,
        forced_retires=forced_retires,
        probabilistic_resets=probabilistic_resets,
        target_turn_completions=target_turn_completions,
        restart_splits=restart_splits,
        retire_fraction=round(forced_retires / max(total, 1), 4),
        reset_fraction=round(probabilistic_resets / max(total, 1), 4),
        target_turn_fraction=round(target_turn_completions / max(total, 1), 4),
        restart_split_fraction=round(restart_splits / max(total, 1), 4),
        final_context_utilization=percentile_stats(fcu_arr),
    )

    return QualityReport(
        config_summary=config_summary(config),
        observed_vs_target=observed_vs_target,
        session_stats=session_stats,
        session_end_stats=session_end_stats,
    )
