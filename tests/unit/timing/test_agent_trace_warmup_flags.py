# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI flag tests for Agent Trace Replay warmup parity flags."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.workload_detect import (
    _resolve_emit_warmup,
    _resolve_use_family_sampling,
)
from tests.unit.conftest import make_run_from_cli

DYNAMO_TRACE = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "graph"
    / "adapters"
    / "fixtures"
    / "dynamo_nested"
    / "nested_2_level.jsonl.gz"
)


def _run(**overrides: object):  # type: ignore[return]
    cli = CLIConfig(
        model_names=["m"],
        tokenizer="builtin",
        input_file=str(DYNAMO_TRACE),
        **overrides,  # type: ignore[arg-type]
    )
    return make_run_from_cli(cli)


@pytest.mark.parametrize(
    "flag_value, expected",
    [
        param(True, True, id="use-family-sampling-true"),
        param(False, False, id="use-family-sampling-false"),
    ],
)  # fmt: skip
def test_graph_use_family_sampling_flag_propagates(
    flag_value: bool, expected: bool
) -> None:
    """--graph-use-family-sampling / --no-graph-use-family-sampling must reach resolver."""
    run = _run(graph_use_family_sampling=flag_value)
    assert _resolve_use_family_sampling(run) == expected


def test_graph_use_family_sampling_defaults_to_true() -> None:
    """Default (no flag) must be True — Agent Trace Replay per-family sampling is on by default."""
    run = _run()
    assert _resolve_use_family_sampling(run) is True


@pytest.mark.parametrize(
    "flag_value, expected",
    [
        param(True, True, id="emit-warmup-true"),
        param(False, False, id="emit-warmup-false"),
    ],
)  # fmt: skip
def test_graph_emit_warmup_flag_propagates(flag_value: bool, expected: bool) -> None:
    """--graph-emit-warmup must reach resolver."""
    run = _run(graph_emit_warmup=flag_value)
    assert _resolve_emit_warmup(run) == expected


def test_graph_emit_warmup_defaults_to_false() -> None:
    """Default (no flag) must be False — warmup is opt-in."""
    run = _run()
    assert _resolve_emit_warmup(run) is False
