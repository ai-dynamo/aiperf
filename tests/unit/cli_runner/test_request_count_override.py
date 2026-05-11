# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: ``--request-count`` overrides YAML ``phases[*].requests``.

Round-2 R2-H3 reproduced this end-to-end: with ``--config minimal.yaml``
and ``--request-count 10``, the CLI flag silently no-opped because the
YAML+CLI resolver only built section-level overrides and never overlaid
loadgen-derived values onto the YAML's profiling phase.

The fix in ``aiperf.config.flags.resolver._apply_phase_loadgen_overrides``
walks the merged envelope and writes loadgen fields onto the phase named
``profiling`` (or the sole non-warmup entry). This test locks that in.
"""

from __future__ import annotations

import pathlib

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config

TEMPLATES_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "config"
    / "templates"
)


def _profiling_requests(cfg) -> int | None:  # noqa: ANN001
    for phase in cfg.benchmark.phases:
        if phase.name == "profiling":
            return getattr(phase, "requests", None)
    return None


def _warmup_requests(cfg) -> int | None:  # noqa: ANN001
    for phase in cfg.benchmark.phases:
        if phase.name == "warmup":
            return getattr(phase, "requests", None)
    return None


def test_request_count_overrides_template_phases_requests() -> None:
    """``--request-count 10`` with ``minimal.yaml`` overrides the YAML's 100."""
    user = CLIConfig(**CLIConfig(request_count=10).model_dump(exclude_unset=True))
    cfg = resolve_config(user, TEMPLATES_DIR / "minimal.yaml")
    assert _profiling_requests(cfg) == 10


def test_request_count_does_not_clobber_warmup_phase() -> None:
    """``--request-count`` targets the profiling phase only; warmup is preserved."""
    user = CLIConfig(**CLIConfig(request_count=10).model_dump(exclude_unset=True))
    cfg = resolve_config(user, TEMPLATES_DIR / "warmup_profiling.yaml")
    # Warmup keeps its YAML-supplied count; only profiling is overridden.
    assert _warmup_requests(cfg) is not None
    assert _warmup_requests(cfg) != 10
    assert _profiling_requests(cfg) == 10


def test_no_loadgen_override_leaves_yaml_intact() -> None:
    """When the user passes no loadgen flags, the YAML's phases stand."""
    user = CLIConfig()
    cfg = resolve_config(user, TEMPLATES_DIR / "minimal.yaml")
    # minimal.yaml ships requests=100.
    assert _profiling_requests(cfg) == 100


def test_concurrency_override_targets_profiling_phase() -> None:
    """The same overlay rule applies to ``--concurrency``."""
    user = CLIConfig(**CLIConfig(concurrency=99).model_dump(exclude_unset=True))
    cfg = resolve_config(user, TEMPLATES_DIR / "minimal.yaml")
    for phase in cfg.benchmark.phases:
        if phase.name == "profiling":
            assert getattr(phase, "concurrency", None) == 99
