# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for ``--search-space`` dimensions vs. the resolved phase shape.

``--search-space`` shape inference only runs when the profiling phase is built
from CLI flags. A config that authors its own ``phases:`` keeps its own
discriminator, so a dimension may name a field the resolved phase type does not
declare. That path is syntactically valid, so before this guard nothing caught
it until the planner wrote its first sampled value and Pydantic raised
``extra_forbidden`` mid-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.errors import ConfigurationError

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

_SEARCH_FLAGS = {
    "search_metric": "ttft",
    "search_direction": "minimize",
    "search_max_iterations": 20,
}


def _yaml(phase_type: str, field_line: str) -> str:
    return (
        _PREAMBLE
        + f"""\
  phases:
    - name: measured
      kind: profiling
      type: {phase_type}
      {field_line}
      requests: 100
"""
    )


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "dimension, field_name, declared_by",
    [
        param("users:1,10:int", "users", "user_centric", id="users_on_concurrency"),
        param("smoothness:0.5,2.0", "smoothness", "gamma", id="smoothness_on_concurrency"),
    ],
)  # fmt: skip
def test_search_space_field_absent_from_phase_type_raises(
    tmp_path: Path, dimension: str, field_name: str, declared_by: str
) -> None:
    """A dimension the resolved phase cannot accept must fail at resolution."""
    config_file = _write(tmp_path, _yaml("concurrency", "concurrency: 8"))

    with pytest.raises(ConfigurationError) as excinfo:
        resolve_config(
            CLIConfig(search_space=[dimension], **_SEARCH_FLAGS), config_file
        )

    message = str(excinfo.value)
    assert field_name in message
    assert "concurrency" in message
    assert declared_by in message


def test_search_space_field_present_on_phase_type_resolves(tmp_path: Path) -> None:
    """A dimension the phase does declare must resolve untouched."""
    config_file = _write(tmp_path, _yaml("user_centric", "users: 4\n      rate: 1.0"))

    config = resolve_config(
        CLIConfig(search_space=["users:1,10:int"], **_SEARCH_FLAGS), config_file
    )

    assert config.benchmark.phases[0].users == 4
    assert [d.path for d in config.sweep.search_space] == ["phases.profiling.users"]


def test_search_space_nested_subfield_is_not_checked(tmp_path: Path) -> None:
    """A deeper path targets a sub-model, whose own validation owns the field."""
    config_file = _write(tmp_path, _yaml("concurrency", "concurrency: 8"))

    config = resolve_config(
        CLIConfig(
            search_space=["phases.profiling.cancellation.rate:0.1,0.5"], **_SEARCH_FLAGS
        ),
        config_file,
    )

    assert config.benchmark.phases[0].type == "concurrency"


def test_search_space_unresolvable_phase_selector_is_skipped(tmp_path: Path) -> None:
    """An unknown phase selector is left to the existing sweep-path handling."""
    config_file = _write(tmp_path, _yaml("concurrency", "concurrency: 8"))

    config = resolve_config(
        CLIConfig(search_space=["phases.nosuchphase.users:1,10:int"], **_SEARCH_FLAGS),
        config_file,
    )

    assert config.benchmark.phases[0].type == "concurrency"
