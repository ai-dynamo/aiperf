# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: ``--max-context-length`` has a single owner per workload kind.

Graph workloads consume the value via ``synthesis.max_context_length``
(``workload_detect._resolve_graph_max_context``). Writing the top-level weka
field too double-owned the value and made a graph run that also named an
explicit non-weka ``--custom-dataset-type`` fail with the weka-only
``ValueError``. Weka runs still get the top-level field.
"""

from __future__ import annotations

import pathlib

import pytest

from aiperf.config.flags import CLIConfig
from aiperf.config.flags._converter_dataset import build_dataset


def _cli(**kwargs) -> CLIConfig:
    return CLIConfig(**kwargs)


@pytest.fixture
def trace(tmp_path: pathlib.Path) -> str:
    path = tmp_path / "trace.jsonl"
    path.write_text("{}\n")
    return str(path)


def test_graph_run_routes_max_context_length_to_synthesis_only(trace: str) -> None:
    """A graph workload gets synthesis.max_context_length and no top-level field."""
    d = build_dataset(
        _cli(
            input_file=trace,
            graph_format="dynamo_trace",
            max_context_length=4096,
        )
    )
    assert d["synthesis"]["max_context_length"] == 4096
    assert "max_context_length" not in d


def test_graph_run_with_explicit_non_weka_type_is_not_rejected(trace: str) -> None:
    """The weka-only guard no longer fires for graph workloads."""
    d = build_dataset(
        _cli(
            input_file=trace,
            graph_format="dynamo_trace",
            custom_dataset_type="mooncake_trace",
            max_context_length=4096,
        )
    )
    assert d["synthesis"]["max_context_length"] == 4096


def test_weka_run_still_gets_top_level_max_context_length(trace: str) -> None:
    """Non-graph weka replay keeps the top-level filter field."""
    d = build_dataset(
        _cli(
            input_file=trace,
            custom_dataset_type="weka_trace",
            max_context_length=4096,
        )
    )
    assert d["max_context_length"] == 4096


def test_non_graph_non_weka_run_still_rejected(trace: str) -> None:
    """The loud rejection is preserved for provably non-weka, non-graph runs."""
    with pytest.raises(ValueError, match="only applies to Weka trace replay"):
        build_dataset(
            _cli(
                input_file=trace,
                custom_dataset_type="mooncake_trace",
                max_context_length=4096,
            )
        )
