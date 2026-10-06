# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep-family CLI flags combined with ``--config``.

The CLI-only path (``convert_cli_to_aiperf``) is the reference behavior; each
test pins one way the ``--config`` path used to diverge from it, or one
companion a flag cannot mean anything without.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from aiperf.config import AIPerfConfig
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config

_PLAIN_YAML = textwrap.dedent("""\
    schemaVersion: "2.0"
    benchmark:
      model: test-model
      endpoint:
        url: http://localhost:8000
        streaming: true
      datasets:
        - name: main
          type: synthetic
          entries: 16
      phases:
        type: concurrency
        concurrency: 1
        requests: 5
""")

_SWEEP_BLOCK = textwrap.dedent("""\
    sweep:
      type: grid
      parameters:
        phases.profiling.concurrency: [1, 2]
""")


def _resolve(
    tmp_path: Path, *, yaml_text: str = _PLAIN_YAML, **flags: Any
) -> AIPerfConfig:
    path = tmp_path / "base.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return resolve_config(CLIConfig(**flags), path)


def _sweep(config: AIPerfConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude_none=True)["sweep"]


# --- recipe output ----------------------------------------------------------


def test_grid_recipe_keeps_recipe_name(tmp_path: Path) -> None:
    sweep = _sweep(_resolve(tmp_path, search_recipe="concurrency-ramp"))
    assert sweep["type"] == "grid"
    assert sweep["recipe_name"] == "concurrency-ramp"


def test_scenario_recipe_emits_its_scenarios(tmp_path: Path) -> None:
    config = _resolve(
        tmp_path,
        search_recipe="pareto-sweep",
        isl_osl_pairs="128/128,256/256",
        concurrency=[1, 4],
    )
    sweep = _sweep(config)
    assert sweep["type"] == "scenarios"
    assert sweep["recipe_name"] == "pareto-sweep"
    assert "parameters" not in sweep
    assert [run["name"] for run in sweep["runs"]] == [
        "shape_128_128_c1",
        "shape_128_128_c4",
        "shape_256_256_c1",
        "shape_256_256_c4",
    ]
    assert config._raw_envelope["sweep"]["type"] == "scenarios"


def test_grid_recipe_plus_magic_list_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="mutually exclusive with magic-list flags"):
        _resolve(tmp_path, search_recipe="concurrency-ramp", concurrency=[1, 2])
