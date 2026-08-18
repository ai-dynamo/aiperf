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
