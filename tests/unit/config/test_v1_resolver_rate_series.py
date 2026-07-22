# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.plugin.enums import PhaseType


def test_yaml_native_inline_rate_series_points(tmp_path: Path) -> None:
    cfg_file = tmp_path / "rate_series.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        benchmark:
          models:
            - test-model
          endpoint:
            urls:
              - http://localhost:8000/v1/chat/completions
          datasets:
            - name: default
              type: synthetic
          phases:
            - name: profiling
              type: poisson
              requests: 10
              rateSeries:
                points:
                  - timeS: 0
                    qps: 5
                  - timeS: 10
                    qps: 15
        """),
        encoding="utf-8",
    )

    config = resolve_config(CLIConfig(), cfg_file)
    phase = config.benchmark.phases[0]

    assert phase.rate is None
    assert phase.rate_series is not None
    assert [(p.time_s, p.qps) for p in phase.rate_series.points] == [
        (0.0, 5.0),
        (10.0, 15.0),
    ]


def test_cli_request_rate_series_overrides_yaml_phase(tmp_path: Path) -> None:
    cfg_file = tmp_path / "base_rate.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        benchmark:
          models:
            - test-model
          endpoint:
            urls:
              - http://localhost:8000/v1/chat/completions
          datasets:
            - name: default
              type: synthetic
          phases:
            - name: profiling
              type: poisson
              requests: 10
              rate: 1
        """),
        encoding="utf-8",
    )
    series_path = tmp_path / "rate.json"
    series_path.write_text(
        '{"points":[{"time_s":0,"qps":4},{"time_s":10,"qps":12}]}',
        encoding="utf-8",
    )
    user = CLIConfig(
        request_rate_series=series_path,
        arrival_pattern="constant",
        request_count=10,
    )

    config = resolve_config(user, cfg_file)
    phase = next(p for p in config.benchmark.phases if p.name == "profiling")

    assert phase.type == PhaseType.CONSTANT
    assert phase.rate is None
    assert phase.rate_series is not None
    assert phase.rate_series.points[1].qps == 12.0


def test_cli_request_rate_rejects_yaml_rate_series(tmp_path: Path) -> None:
    cfg_file = tmp_path / "series_only.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        benchmark:
          models:
            - test-model
          endpoint:
            urls:
              - http://localhost:8000/v1/chat/completions
          datasets:
            - name: default
              type: synthetic
          phases:
            - name: profiling
              type: poisson
              requests: 10
              rateSeries:
                points:
                  - timeS: 0
                    qps: 5
                  - timeS: 10
                    qps: 15
        """),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="mutually exclusive"):
        resolve_config(CLIConfig(request_rate=50), cfg_file)
