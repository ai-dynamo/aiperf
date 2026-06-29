# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Issue #13 regression: ``-f base.yaml --streaming`` must reach the recipe.

When YAML pins ``endpoint.streaming: false`` and the user passes
``--streaming`` on the CLI to override it, the recipe's
``require_streaming()`` check must see the merged value (True), not the
stale YAML value. The resolver previously built its
``base_config.benchmark`` from YAML alone before invoking
``expand_search_recipe``, so the recipe rejected the call.
"""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config

_YAML_STREAMING_OFF = textwrap.dedent("""\
benchmark:
  models:
    - test-model
  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    streaming: false
  datasets:
    - name: default
      type: synthetic
      entries: 100
      prompts:
        isl: 128
        osl: 64
  phases:
    - name: profiling
      type: concurrency
      requests: 10
      concurrency: 1
""")


def _write_yaml(tmp_path: Path) -> Path:
    cfg_file = tmp_path / "base.yaml"
    cfg_file.write_text(_YAML_STREAMING_OFF)
    return cfg_file


def test_cli_streaming_override_reaches_recipe(tmp_path: Path) -> None:
    """``-f streaming-off.yaml --search-recipe prefill-ttft-curve --streaming``
    must succeed: the CLI ``--streaming`` overrides the YAML ``streaming: false``
    BEFORE the resolver hands the BenchmarkConfig to ``expand_search_recipe``.
    """
    cfg_file = _write_yaml(tmp_path)
    user = CLIConfig(
        streaming=True,
        **CLIConfig(concurrency=1, request_count=10).model_dump(exclude_unset=True),
        search_recipe="prefill-ttft-curve",
    )

    config = resolve_config(user, cfg_file)

    # ``prefill-ttft-curve`` is a streaming-only recipe; reaching here means
    # ``require_streaming`` saw the merged ``streaming=True``.
    assert config.benchmark.endpoint.streaming is True
    assert config.sweep is not None


def test_yaml_streaming_off_without_cli_override_still_rejects(
    tmp_path: Path,
) -> None:
    """Counterpart sanity check: when the user does NOT pass ``--streaming``,
    the recipe's check must still hard-reject the YAML's ``streaming: false``.
    """
    cfg_file = _write_yaml(tmp_path)
    user = CLIConfig(
        **CLIConfig(concurrency=1, request_count=10).model_dump(exclude_unset=True),
        search_recipe="prefill-ttft-curve",
    )

    with pytest.raises(ValueError, match="requires --streaming"):
        resolve_config(user, cfg_file)


def test_yaml_cli_magic_list_promotes_to_sweep(tmp_path: Path) -> None:
    cfg_file = _write_yaml(tmp_path)
    user = CLIConfig(concurrency=[1, 2], request_count=10)

    config = resolve_config(user, cfg_file)

    assert config.sweep is not None
    assert config.sweep.parameters["phases.profiling.concurrency"] == [1, 2]
    assert config.benchmark.phases[0].concurrency == 1


def test_yaml_cli_dataset_magic_list_targets_existing_dataset(tmp_path: Path) -> None:
    cfg_file = _write_yaml(tmp_path)
    user = CLIConfig(prompt_input_tokens_mean=[128, 256], request_count=10)

    config = resolve_config(user, cfg_file)

    assert config.sweep is not None
    assert config.sweep.parameters["datasets.default.prompts.isl.mean"] == [128, 256]
    assert "datasets.main.prompts.isl.mean" not in config.sweep.parameters


_YAML_ADVANCED_ADAPTIVE = textwrap.dedent("""\
benchmark:
  models:
    - test-model
  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    streaming: true
  datasets:
    - name: default
      type: synthetic
      entries: 100
      prompts:
        isl: 128
        osl: 64
  phases:
    - name: profiling
      type: concurrency
      duration: 60
      concurrency: 8
      sla:
        request_latency:
          p95:
            le: 30000
      adaptive_scale:
        enabled: false
        min_concurrency: 2
        min_completed_requests: 3
        sustain_duration: 20
        assessment_period: 5
        strategy:
          type: ramp_until_fail
          step_policy: fixed_percent_step
          step_percent: 50
""")


def test_basic_adaptive_cli_overrides_preserve_advanced_yaml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "adaptive.yaml"
    cfg_file.write_text(_YAML_ADVANCED_ADAPTIVE)
    user = CLIConfig(
        adaptive_scale=True,
        adaptive_sustain_duration=40,
        adaptive_assessment_period=10,
        concurrency=16,
        adaptive_scale_sla=["request_latency:p95:le:20000"],
    )

    config = resolve_config(user, cfg_file)
    phase = config.benchmark.phases[0]

    assert phase.adaptive_scale is True
    assert phase.concurrency == 16
    assert phase.adaptive_sustain_duration == 40
    assert phase.adaptive_assessment_period == 10
    assert phase.sla[0].threshold == 20000

    assert phase.adaptive_scale_min_concurrency == 2
    assert phase.adaptive_min_completed_requests == 3
    assert phase.adaptive_scale_step_policy == "fixed_percent_step"
    assert phase.adaptive_scale_step_percent == 50


async def test_adaptive_cli_sla_parse_error_names_adaptive_flag(tmp_path: Path) -> None:
    cfg_file = tmp_path / "adaptive.yaml"
    await asyncio.to_thread(cfg_file.write_text, _YAML_ADVANCED_ADAPTIVE)
    user = CLIConfig(adaptive_scale_sla=["bad"])

    with pytest.raises(TypeError) as exc_info:
        resolve_config(user, cfg_file)

    message = str(exc_info.value)
    assert "--adaptive-scale-sla" in message
    assert "--search-sla" not in message
