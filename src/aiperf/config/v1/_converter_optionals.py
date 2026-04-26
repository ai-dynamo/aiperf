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

    The v1 ``LoadGeneratorConfig`` carries multi-run knobs flat alongside the
    rest of load-generator config (matching origin/main). Returns ``None``
    when ``user.loadgen`` is unset or no multi-run fields were explicitly set.
    """
    lg = user.loadgen
    if lg is None or not lg.model_fields_set:
        return None
    # field-on-loadgen -> output-key
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
    }
    out: dict[str, Any] = {}
    for field, key in mapping.items():
        if field in lg.model_fields_set:
            out[key] = getattr(lg, field)
    return out or None
