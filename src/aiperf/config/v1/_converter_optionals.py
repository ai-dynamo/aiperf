# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional-section builders for the v1 -> v2 UserConfig converter.

Each builder inspects a nested section on the v1 ``UserConfig`` and, when at
least one field was explicitly set by the user, returns a dict shaped for
``AIPerfConfig`` consumption. When the section is absent or no fields were
set, the builder returns ``None`` so the top-level converter can omit the
section cleanly rather than emitting empty sub-objects.

Mirrors the section-builder logic in ``aiperf.config._cli_sections`` for the
flat CLIModel input, rerouted to read from nested ``UserConfig`` sub-models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.v1 import UserConfig


def build_tokenizer(user: UserConfig) -> dict[str, Any] | None:
    """Build the tokenizer section dict from explicitly-set v1 fields.

    Returns ``None`` when ``user.tokenizer`` is unset or has no explicitly
    populated fields (so the converter skips the section entirely).
    """
    tok = user.tokenizer
    if tok is None or not tok.model_fields_set:
        return None
    out: dict[str, Any] = {}
    if "name" in tok.model_fields_set:
        out["name"] = tok.name
    if "revision" in tok.model_fields_set:
        out["revision"] = tok.revision
    if "trust_remote_code" in tok.model_fields_set:
        out["trust_remote_code"] = tok.trust_remote_code
    return out or None


def build_accuracy(user: UserConfig) -> dict[str, Any] | None:
    """Build the accuracy section dict from explicitly-set v1 fields.

    Returns ``None`` when ``user.accuracy`` is unset or has no explicitly
    populated fields.
    """
    acc = user.accuracy
    if acc is None or not acc.model_fields_set:
        return None
    keys = (
        "benchmark",
        "tasks",
        "n_shots",
        "enable_cot",
        "grader",
        "system_prompt",
        "verbose",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if key in acc.model_fields_set:
            out[key] = getattr(acc, key)
    return out or None


def build_multi_run(user: UserConfig) -> dict[str, Any] | None:
    """Build the multi-run section dict from explicitly-set v1 loadgen fields.

    When --search-* flags are present, builds a typed AdaptiveSearchConfig and
    emits its model_dump() as `out["adaptive_search"]`. MultiRunConfig has
    `extra="forbid"` so the typed field is the only legal carrier.

    Hard-fails if --search-space is set without the required companion flags
    (--search-metric, --search-direction, --search-max-iterations).
    """
    lg = user.loadgen
    if lg is None or not lg.model_fields_set:
        return None
    mapping = {
        "num_profile_runs": "num_runs",
        "profile_run_cooldown_seconds": "cooldown_seconds",
        "confidence_level": "confidence_level",
        "profile_run_disable_warmup_after_first": "disable_warmup_after_first",
        "set_consistent_seed": "set_consistent_seed",
        "convergence_metric": "convergence_metric",
        "convergence_mode": "convergence_mode",
        "convergence_threshold": "convergence_threshold",
        "convergence_stat": "convergence_stat",
        "parameter_sweep_cooldown_seconds": "parameter_sweep_cooldown_seconds",
        "parameter_sweep_same_seed": "parameter_sweep_same_seed",
        "parameter_sweep_mode": "mode",
    }
    out: dict[str, Any] = {}
    for field, key in mapping.items():
        if field in lg.model_fields_set:
            out[key] = getattr(lg, field)
    adaptive_search = _build_adaptive_search(lg)
    if adaptive_search is not None:
        out["adaptive_search"] = adaptive_search
        # --search-* and --convergence-metric (trial-level adaptive early-stop)
        # are conceptually orthogonal but their interaction wasn't designed:
        # the BO orchestrator path silently ignores convergence_metric. Reject
        # explicitly so users don't think trial-level convergence is doing
        # anything during a BO run. Documented in docs/sweeping/bayesian-optimization.md.
        if (
            "convergence_metric" in lg.model_fields_set
            and lg.convergence_metric is not None
        ):
            raise TypeError(
                "--search-* (Bayesian Optimization) is mutually exclusive with "
                "--convergence-metric (trial-level adaptive early-stop). The two "
                "operate at different levels (outer-loop vs. inner-trial) and "
                "their composition is undefined. Drop one of them."
            )
    return out or None


def _build_adaptive_search(lg: Any) -> dict[str, Any] | None:
    """Parse --search-* flags into a model-dumped AdaptiveSearchConfig dict.

    Returns ``None`` when no --search-* flags were set. Raises ``TypeError``
    when the flag combination is invalid (search-space without companions, or
    other --search-* flags without --search-space).
    """
    search_fields = (
        "search_space",
        "search_metric",
        "search_stat",
        "search_direction",
        "search_max_iterations",
        "search_initial_points",
        "search_random_seed",
    )
    search_set = {f for f in search_fields if f in lg.model_fields_set}
    if "search_space" not in search_set:
        if search_set:
            raise TypeError(
                f"--search-* flags {sorted(search_set)} require --search-space."
            )
        return None
    for required, flag in (
        ("search_metric", "--search-metric"),
        ("search_direction", "--search-direction"),
        ("search_max_iterations", "--search-max-iterations"),
    ):
        if required not in search_set:
            raise TypeError(
                f"--search-space requires {flag} (companion flag missing). "
                "See docs/sweeping/bayesian-optimization.md for examples."
            )
    # Done here (not later in build_benchmark_plan) so MultiRunConfig
    # validation catches structural errors early at the v1->v2 boundary.
    from aiperf.config.adaptive_search import AdaptiveSearchConfig
    from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
    from aiperf.orchestrator.search_planner.parsing import parse_search_space

    ol_kwargs: dict[str, Any] = dict(
        algorithm="bayes",
        search_space=parse_search_space(lg.search_space),
        objective_metric=lg.search_metric,
        objective_stat=lg.search_stat or "avg",
        objective_direction=OptimizationDirection(lg.search_direction),
        max_iterations=lg.search_max_iterations,
    )
    if "search_initial_points" in search_set and lg.search_initial_points is not None:
        ol_kwargs["n_initial_points"] = lg.search_initial_points
    if "search_random_seed" in search_set and lg.search_random_seed is not None:
        ol_kwargs["random_seed"] = lg.search_random_seed
    return AdaptiveSearchConfig(**ol_kwargs).model_dump(mode="json")
