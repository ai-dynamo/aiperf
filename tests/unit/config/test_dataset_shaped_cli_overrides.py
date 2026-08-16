# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: dataset-shaped CLI flags survive the ``-f config.yaml`` path.

``build_cli_overrides`` never calls ``build_dataset``, so dataset-shaped
members of ``INPUT_FIELDS`` (``--graph-format``,
``--trace-idle-gap-cap-seconds``) had no overlay under a YAML config and were
silently dropped, contradicting the documented CLI-over-YAML contract. The
fix is ``resolver._apply_dataset_shaped_overrides``.
"""

from __future__ import annotations

import pathlib

import pytest

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config

_YAML = """\
schemaVersion: "2.0"
benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000
  dataset:
    type: file
    path: {path}
  phases:
    type: concurrency
    concurrency: 1
    requests: 1
"""


@pytest.fixture
def yaml_config(tmp_path: pathlib.Path) -> pathlib.Path:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_YAML.format(path=trace))
    return cfg


def _dataset(cfg):  # noqa: ANN001
    return cfg.benchmark.datasets[0]


def test_graph_format_cli_flag_overlays_yaml_dataset(
    yaml_config: pathlib.Path,
) -> None:
    """``--graph-format`` reaches the YAML-supplied dataset."""
    user = CLIConfig(
        **CLIConfig(graph_format="dynamo_trace").model_dump(exclude_unset=True)
    )
    cfg = resolve_config(user, yaml_config)
    assert str(_dataset(cfg).graph_format) == "dynamo_trace"


def test_trace_idle_gap_cap_cli_flag_overlays_yaml_dataset(
    yaml_config: pathlib.Path,
) -> None:
    """``--trace-idle-gap-cap-seconds`` reaches the YAML dataset."""
    user = CLIConfig(
        **CLIConfig(trace_idle_gap_cap_seconds=5.0).model_dump(exclude_unset=True)
    )
    cfg = resolve_config(user, yaml_config)
    assert _dataset(cfg).trace_idle_gap_cap_seconds == 5.0


def test_graph_tool_persistent_session_cli_flag_overlays_yaml_dataset(
    yaml_config: pathlib.Path,
) -> None:
    user = CLIConfig(
        **CLIConfig(graph_tool_persistent_session=True).model_dump(exclude_unset=True)
    )

    cfg = resolve_config(user, yaml_config)

    assert _dataset(cfg).graph_tool_persistent_session is True


def test_dataset_shaped_flags_unset_leave_yaml_untouched(
    yaml_config: pathlib.Path,
) -> None:
    """No overlay happens when neither flag is explicitly set."""
    cfg = resolve_config(CLIConfig(), yaml_config)
    assert _dataset(cfg).graph_format is None
