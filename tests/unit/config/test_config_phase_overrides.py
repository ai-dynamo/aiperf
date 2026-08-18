# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-shaping CLI flags must survive ``--config``.

The resolver overlaid only the flat loadgen fields onto the profiling phase.
The ramp durations, request cancellation, and arrival smoothness were built
solely by the CLI-only converter, so under ``--config`` they were dropped --
and then, once the classification gate landed, rejected outright.

They are routed here by reusing the converter's own helpers rather than
restating the mapping, so the two paths cannot drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.errors import ConfigurationError


def cli(**kwargs: object) -> CLIConfig:
    return CLIConfig(**CLIConfig(**kwargs).model_dump(exclude_unset=True))  # type: ignore[arg-type]


def profiling(cfg):
    return next(p for p in cfg.benchmark.phases if p.name == "profiling")


@pytest.fixture
def concurrency_yaml(tmp_path: Path) -> Path:
    cfg = tmp_path / "conc.yaml"
    cfg.write_text(
        """\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
  phases:
    type: concurrency
    concurrency: 4
    requests: 5
"""
    )
    return cfg


@pytest.fixture
def gamma_yaml(tmp_path: Path) -> Path:
    cfg = tmp_path / "gamma.yaml"
    cfg.write_text(
        """\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
  phases:
    type: gamma
    rate: 10
    requests: 5
"""
    )
    return cfg


# ---------------------------------------------------------------------------
# Ramps
# ---------------------------------------------------------------------------


def test_concurrency_ramp_duration_routes(concurrency_yaml: Path) -> None:
    resolved = profiling(
        resolve_config(cli(concurrency_ramp_duration=30), concurrency_yaml)
    )
    assert resolved.concurrency_ramp is not None
    assert resolved.concurrency_ramp.duration == 30


def test_prefill_concurrency_ramp_duration_routes(concurrency_yaml: Path) -> None:
    resolved = profiling(
        resolve_config(cli(prefill_concurrency_ramp_duration=15), concurrency_yaml)
    )
    assert resolved.prefill_ramp is not None
    assert resolved.prefill_ramp.duration == 15


def test_ramp_does_not_disturb_other_phase_values(concurrency_yaml: Path) -> None:
    resolved = profiling(
        resolve_config(cli(concurrency_ramp_duration=30), concurrency_yaml)
    )
    assert resolved.concurrency == 4
    assert resolved.requests == 5


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_request_cancellation_rate_routes(concurrency_yaml: Path) -> None:
    resolved = profiling(
        resolve_config(cli(request_cancellation_rate=25.0), concurrency_yaml)
    )
    assert resolved.cancellation is not None
    assert resolved.cancellation.rate == 25.0


def test_request_cancellation_delay_routes_with_rate(concurrency_yaml: Path) -> None:
    resolved = profiling(
        resolve_config(
            cli(request_cancellation_rate=25.0, request_cancellation_delay=2.0),
            concurrency_yaml,
        )
    )
    assert resolved.cancellation.delay == 2.0


def test_cancellation_delay_without_rate_still_errors(concurrency_yaml: Path) -> None:
    """The converter's dependency guard must hold on this path too."""
    with pytest.raises(ValueError, match="requires --request-cancellation-rate"):
        resolve_config(cli(request_cancellation_delay=2.0), concurrency_yaml)


# ---------------------------------------------------------------------------
# Arrival smoothness
# ---------------------------------------------------------------------------


def test_arrival_smoothness_routes_on_gamma_phase(gamma_yaml: Path) -> None:
    resolved = profiling(resolve_config(cli(arrival_smoothness=0.5), gamma_yaml))
    assert resolved.smoothness == 0.5


def test_arrival_smoothness_rejected_on_non_gamma_phase(
    concurrency_yaml: Path,
) -> None:
    with pytest.raises(ValueError, match="arrival-smoothness"):
        resolve_config(cli(arrival_smoothness=0.5), concurrency_yaml)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def warmup(cfg):
    return next((p for p in cfg.benchmark.phases if p.name == "warmup"), None)


@pytest.fixture
def warmup_yaml(tmp_path: Path) -> Path:
    """A config that already declares a warmup phase."""
    cfg = tmp_path / "warmup.yaml"
    cfg.write_text(
        """\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
  phases:
    - name: warmup
      kind: warmup
      type: concurrency
      concurrency: 2
      requests: 3
    - name: profiling
      kind: profiling
      type: concurrency
      concurrency: 4
      requests: 5
"""
    )
    return cfg


def test_warmup_flag_overrides_existing_warmup_phase(warmup_yaml: Path) -> None:
    resolved = resolve_config(cli(warmup_request_count=9), warmup_yaml)
    assert warmup(resolved).requests == 9


def test_warmup_flag_leaves_untouched_warmup_values_alone(warmup_yaml: Path) -> None:
    """Only what the user set may change on the YAML's warmup phase."""
    resolved = resolve_config(cli(warmup_request_count=9), warmup_yaml)
    assert warmup(resolved).concurrency == 2


def test_warmup_flag_does_not_disturb_the_profiling_phase(warmup_yaml: Path) -> None:
    resolved = resolve_config(cli(warmup_request_count=9), warmup_yaml)
    assert profiling(resolved).requests == 5
    assert profiling(resolved).concurrency == 4


def test_secondary_warmup_flag_applies_to_an_existing_phase(
    warmup_yaml: Path,
) -> None:
    """--warmup-concurrency has no trigger of its own.

    On the CLI-only path that means "no warmup phase", so the flag is
    ignored. Here the config file already declares the phase, so the flag
    has somewhere to land and must be applied rather than dropped.
    """
    resolved = resolve_config(cli(warmup_concurrency=7), warmup_yaml)
    assert warmup(resolved).concurrency == 7


def test_warmup_trigger_creates_a_phase_when_yaml_has_none(
    concurrency_yaml: Path,
) -> None:
    """A trigger flag builds the warmup phase, as it does CLI-only."""
    resolved = resolve_config(cli(warmup_request_count=4), concurrency_yaml)
    created = warmup(resolved)
    assert created is not None
    assert created.requests == 4
    assert created.exclude_from_results is True


def test_secondary_warmup_flag_without_a_phase_or_trigger_errors(
    concurrency_yaml: Path,
) -> None:
    """Nowhere to land and nothing to create it: must be loud, not dropped."""
    with pytest.raises(ConfigurationError, match=r"warmup"):
        resolve_config(cli(warmup_concurrency=7), concurrency_yaml)
