# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fidelity regressions for the config-file CLI override pipeline.

Three distinct hazards live here:

1. The rendered envelope (used for Pydantic validation) and the retained
   pre-Jinja envelope (executed by ``build_benchmark_plan``) must agree on
   value-dependent phase-shape decisions, even when the raw envelope still
   carries ``{{ jinja }}`` source strings where the rendered one has values.
2. Ramp CLI duration flags must merge into an existing YAML ramp mapping
   instead of replacing it, for both snake_case and camelCase spellings.
3. ``--network-latency-automatic`` must clear a YAML-supplied ``meanMs`` so
   active probing actually runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.plan import build_benchmark_plan
from aiperf.plugin.enums import PhaseType, RampType

_PREAMBLE = """\
benchmark:
  models:
    items:
      - name: yaml-model
  endpoint:
    urls: [http://localhost:8000]
  datasets:
    - name: workload
      type: synthetic
"""

_TEMPLATED_USER_CENTRIC_SWEEP = (
    """\
variables:
  ptype: user_centric
sweep:
  type: grid
  parameters:
    phases.measured.requests: [100, 200]
"""
    + _PREAMBLE
    + """\
  phases:
    - name: measured
      kind: profiling
      type: "{{ ptype }}"
      users: 4
      rate: 1.0
      requests: 100
"""
)

_TEMPLATED_FIXED_SCHEDULE_SWEEP = (
    """\
variables:
  ptype: fixed_schedule
sweep:
  type: grid
  parameters:
    phases.measured.requests: [10, 20]
"""
    + _PREAMBLE
    + """\
  phases:
    - name: measured
      kind: profiling
      type: "{{ ptype }}"
      requests: 10
"""
)

_NETWORK_LATENCY_MEAN_YAML = (
    _PREAMBLE
    + """\
  phases:
    - name: measured
      kind: profiling
      type: concurrency
      concurrency: 8
      requests: 10
  networkLatency: {enabled: true, meanMs: 5.0}
"""
)


def _ramp_yaml(ramp_key: str) -> str:
    return (
        _PREAMBLE
        + f"""\
  phases:
    - name: measured
      kind: profiling
      type: concurrency
      concurrency: 8
      requests: 10
      {ramp_key}: {{duration: 10, strategy: exponential}}
"""
    )


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_request_rate_over_templated_user_centric_phase_keeps_executed_plan_closed_loop(
    tmp_path: Path,
) -> None:
    """The executed plan must match the validated config's phase shape."""
    config_file = _write(tmp_path, _TEMPLATED_USER_CENTRIC_SWEEP)

    config = resolve_config(CLIConfig(request_rate=9.0), config_file)
    plan = build_benchmark_plan(config)

    assert config.benchmark.phases[0].type == PhaseType.USER_CENTRIC
    for executed in plan.configs:
        phase = executed.phases[0]
        assert phase.type == PhaseType.USER_CENTRIC
        assert phase.users == 4
        assert phase.rate == 9.0


def test_fixed_schedule_offset_over_templated_phase_type_does_not_raise(
    tmp_path: Path,
) -> None:
    """A Jinja-sourced ``type`` must not fail the raw-envelope guard."""
    config_file = _write(tmp_path, _TEMPLATED_FIXED_SCHEDULE_SWEEP)

    config = resolve_config(CLIConfig(fixed_schedule_start_offset=5), config_file)
    plan = build_benchmark_plan(config)

    assert config.benchmark.phases[0].start_offset == 5
    for executed in plan.configs:
        assert executed.phases[0].type == PhaseType.FIXED_SCHEDULE
        assert executed.phases[0].start_offset == 5


@pytest.mark.parametrize(
    "ramp_key",
    [
        param("concurrencyRamp", id="camel-case"),
        param("concurrency_ramp", id="snake-case"),
    ],
)  # fmt: skip
def test_ramp_duration_flag_preserves_sibling_strategy(
    tmp_path: Path, ramp_key: str
) -> None:
    """``--concurrency-ramp-duration`` must not drop the YAML ``strategy``."""
    config_file = _write(tmp_path, _ramp_yaml(ramp_key))

    config = resolve_config(CLIConfig(concurrency_ramp_duration=20.0), config_file)

    ramp = config.benchmark.phases[0].concurrency_ramp
    assert ramp.duration == 20.0
    assert ramp.strategy == RampType.EXPONENTIAL


def test_network_latency_automatic_clears_yaml_mean_ms(tmp_path: Path) -> None:
    """``--network-latency-automatic`` must re-enable active probing."""
    config_file = _write(tmp_path, _NETWORK_LATENCY_MEAN_YAML)

    config = resolve_config(CLIConfig(network_latency_automatic=True), config_file)

    network_latency = config.benchmark.network_latency
    assert network_latency.enabled is True
    assert network_latency.mean_ms is None
    assert network_latency.should_probe is True
