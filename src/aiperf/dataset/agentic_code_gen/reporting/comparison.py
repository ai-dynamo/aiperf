# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shareable text comparisons for generated Agentic Code datasets."""

from __future__ import annotations

from aiperf.dataset.agentic_code_gen.models import PercentileStats


def render_comparison_text(
    quality: dict,
    *,
    session_duration_stats: PercentileStats | None = None,
    prefill_tps: float = 20_000,
    decode_tps: float = 60,
    target_p05: float | None = 30,
    target_p99: float | None = 3_750,
) -> str:
    """Render a shareable target-vs-dataset comparison from quality.json data."""
    cfg = quality["config_summary"]
    ovt = quality["observed_vs_target"]
    sess = quality["session_stats"]
    ends = quality["session_end_stats"]
    num_sessions = ends["total_sessions"]
    total_turns = ovt.get("generation_length", {}).get("observed", {}).get("count", 0)

    lines: list[str] = []
    w = lines.append

    w("Agentic Code Session Profile: Target vs Synthesized Dataset")
    w("=" * 60)
    w("")
    w(
        f"{num_sessions:,} sessions | {total_turns:,} turns | block_size={cfg['block_size']}"
    )
    w("")
    w(f"{'':40s} {'Target':>12s} {'Dataset':>12s} {'Error':>8s}")
    w("-" * 76)

    def row(
        row_label: str,
        target_val: float | None,
        obs_val: float | None,
        pct_err: float | None = None,
    ) -> None:
        target = f"{target_val:>12,.0f}" if target_val is not None else f"{'-':>12s}"
        observed = f"{obs_val:>12,.1f}" if obs_val is not None else f"{'-':>12s}"
        error = ""
        if pct_err is not None:
            sign = (
                "+"
                if obs_val is None
                or target_val is None
                or abs(pct_err) == 0.0
                or obs_val >= target_val
                else "-"
            )
            error = f"  {sign}{abs(pct_err):.1f}%"
        w(f"  {row_label:38s}{target}{observed}{error}")

    def metric_block(label: str, key: str) -> None:
        w(label)
        metric = ovt.get(key, {})
        observed = metric.get("observed", {})
        row(
            "mean",
            metric.get("target_mean"),
            observed.get("mean"),
            metric.get("pct_error_mean"),
        )
        row(
            "median",
            metric.get("target_median"),
            observed.get("median"),
            metric.get("pct_error_median"),
        )

    metric_block("Initial Context (tokens)", "initial_context")
    w("")

    metric_block("New Tokens Per Turn", "new_tokens_per_turn")
    w("")

    gen = ovt.get("generation_length", {})
    gen_obs = gen.get("observed", {})
    w("Generation Length (tokens)")
    row("mean", gen.get("target_mean"), gen_obs.get("mean"), gen.get("pct_error_mean"))
    row(
        "median",
        gen.get("target_median"),
        gen_obs.get("median"),
        gen.get("pct_error_median"),
    )
    row("p05", target_p05, gen_obs.get("p05"))
    row("p99", target_p99, gen_obs.get("p99"))
    w("")

    w("Prompt")
    w(
        f"  {'max_prompt_tokens':38s}{cfg['max_prompt_tokens']:>12,d}{cfg['max_prompt_tokens']:>12,d}"
    )
    w("")

    w("Additional Dataset Statistics")
    w("-" * 76)

    w("Turns Per Session")
    for label, field in [
        ("mean", "mean"),
        ("median", "median"),
        ("p05", "p05"),
        ("p25", "p25"),
        ("p75", "p75"),
        ("p95", "p95"),
        ("p99", "p99"),
    ]:
        val = sess.get(field, 0)
        w(f"  {label:38s}{'-':>12s}{val:>12.1f}")
    w("")

    delay = ovt.get("inter_turn_delay_ms", {}).get("observed", {})
    if delay:
        w("Inter-Turn Delay (ms)")
        for label, field in [
            ("mean", "mean"),
            ("median", "median"),
            ("p05", "p05"),
            ("p95", "p95"),
        ]:
            val = delay.get(field, 0)
            w(f"  {label:38s}{'-':>12s}{val:>12,.0f}")
        af = cfg.get("inter_turn_delay_agentic_fraction", 0)
        am = cfg.get("inter_turn_delay_agentic_mean_ms", 0) / 1000
        hm = cfg.get("inter_turn_delay_human_mean_ms", 0) / 1000
        w(f"  ({af:.0%} agentic ~{am:.0f}s, {1 - af:.0%} human ~{hm:.0f}s)")
        w("")

    if session_duration_stats:
        w(
            f"Session Duration (estimated @ {prefill_tps:,.0f} prefill tok/s, {decode_tps:,.0f} decode tok/s)"
        )
        for label, val in [
            ("mean", session_duration_stats.mean),
            ("median", session_duration_stats.median),
        ]:
            w(f"  {label:38s}{'-':>12s}{val:>12.1f} min")
        w("")

    w("Session Endings")
    w(
        f"  {'forced retires (hit context limit)':38s}{'-':>12s}{ends['forced_retires']:>12d}"
    )
    w(f"  {'probabilistic resets':38s}{'-':>12s}{ends['probabilistic_resets']:>12d}")
    w(f"  {'restart splits':38s}{'-':>12s}{ends.get('restart_splits', 0):>12d}")

    return "\n".join(lines) + "\n"
