# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: AGENTIC_REPLAY phase flags override YAML-supplied phases."""

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


def _profiling_phase(cfg):  # noqa: ANN001
    for phase in cfg.benchmark.phases:
        if phase.name == "profiling":
            return phase
    return None


def _cli(**kwargs) -> CLIConfig:
    """Build a CLIConfig with only the given fields marked explicitly set."""
    return CLIConfig(**CLIConfig(**kwargs).model_dump(exclude_unset=True))


_AGENTIC_PROFILING_YAML = """\
schemaVersion: "2.0"
benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000
    type: chat
  dataset:
    type: synthetic
    entries: 100
    prompts:
      isl: 512
      osl: 128
  phases:
    type: concurrency
    concurrency: 8
    requests: 100
    timingMode: agentic_replay
"""


def _agentic_yaml(tmp_path: pathlib.Path) -> pathlib.Path:
    """A YAML whose profiling phase resolves to AGENTIC_REPLAY, needed because ``validate_agentic_cache_warmup`` rejects ``--agentic-cache-warmup-duration`` on a non-agentic run."""
    path = tmp_path / "agentic_profiling.yaml"
    path.write_text(_AGENTIC_PROFILING_YAML)
    return path


def test_agentic_cache_warmup_duration_overrides_yaml_profiling_phase(
    tmp_path: pathlib.Path,
) -> None:
    """``--agentic-cache-warmup-duration`` overlays the YAML profiling phase despite not being in ``LOADGEN_FIELDS``."""
    cfg = resolve_config(
        _cli(agentic_cache_warmup_duration=30.0), _agentic_yaml(tmp_path)
    )
    assert _profiling_phase(cfg).agentic_cache_warmup_duration == 30.0


def test_agentic_cache_warmup_request_budget_overrides_yaml_profiling_phase(
    tmp_path: pathlib.Path,
) -> None:
    cfg = resolve_config(
        _cli(
            agentic_cache_warmup_duration=30.0,
            agentic_cache_warmup_requests_per_lane=10,
        ),
        _agentic_yaml(tmp_path),
    )
    phase = _profiling_phase(cfg)
    assert phase.agentic_cache_warmup_duration == 30.0
    assert phase.agentic_cache_warmup_requests_per_lane == 10


def test_agentic_replay_sibling_flags_override_yaml_profiling_phase() -> None:
    """The four sibling agentic-replay phase flags overlay the profiling phase."""
    cfg = resolve_config(
        _cli(
            burst_phase_starts=True,
            failed_request_threshold=0.5,
            trajectory_start_min_ratio=0.1,
            trajectory_start_max_ratio=0.9,
        ),
        TEMPLATES_DIR / "minimal.yaml",
    )
    phase = _profiling_phase(cfg)
    assert phase.burst_phase_starts is True
    assert phase.failed_request_threshold == 0.5
    assert phase.trajectory_start_min_ratio == 0.1
    assert phase.trajectory_start_max_ratio == 0.9


def test_no_agentic_flags_leaves_phase_defaults_intact() -> None:
    """Without the flags, the profiling phase keeps its defaults."""
    cfg = resolve_config(CLIConfig(), TEMPLATES_DIR / "minimal.yaml")
    phase = _profiling_phase(cfg)
    assert phase.agentic_cache_warmup_duration is None
    assert phase.burst_phase_starts is False


_AGENTIC_WARMUP_PROFILING_YAML = """\
schemaVersion: "2.0"
benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000/v1/chat/completions
    type: chat
  dataset:
    type: synthetic
    entries: 100
    prompts:
      isl: 512
      osl: 128
  warmup:
    type: concurrency
    requests: 100
    concurrency: 8
  profiling:
    type: concurrency
    requests: 100
    concurrency: 64
    timingMode: agentic_replay
"""


def test_agentic_warmup_does_not_clobber_warmup_phase(
    tmp_path: pathlib.Path,
) -> None:
    """The overlay targets the profiling phase only; warmup is preserved."""
    path = tmp_path / "agentic_warmup_profiling.yaml"
    path.write_text(_AGENTIC_WARMUP_PROFILING_YAML)
    cfg = resolve_config(_cli(agentic_cache_warmup_duration=30.0), path)
    assert _profiling_phase(cfg).agentic_cache_warmup_duration == 30.0
    for phase in cfg.benchmark.phases:
        if phase.name == "warmup":
            assert getattr(phase, "agentic_cache_warmup_duration", None) is None
