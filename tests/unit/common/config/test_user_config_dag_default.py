# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct graph inputs remain opaque during CLI-to-Config-v2 conversion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.phases import PhaseType

_DIRECT_GRAPH_FORMATS = ("dag_jsonl", "dynamo_trace", "weka_trace")


def _make_cli(**overrides) -> CLIConfig:
    """Build a minimal CLIConfig with endpoint+model, overriding the rest."""
    base = {
        "url": "http://localhost:8000/test",
        "model_names": ["test-model"],
    }
    base.update(overrides)
    return CLIConfig(**base)


def _opaque_graph_file(tmp_path: Path, format_name: str) -> Path:
    path = tmp_path / f"{format_name}.opaque"
    path.write_bytes(b"\xffnot-python-json\x00" + b"x" * (2 * 1024 * 1024))
    return path


@pytest.mark.parametrize("format_name", _DIRECT_GRAPH_FORMATS)
def test_unbounded_direct_graph_uses_ordinary_default_without_opening_payload(
    tmp_path: Path, format_name: str
) -> None:
    graph = _opaque_graph_file(tmp_path, format_name)
    cli = _make_cli(input_file=str(graph), custom_dataset_type=format_name)

    with patch("builtins.open", side_effect=AssertionError("graph payload opened")):
        profiling = build_profiling(cli)

    assert profiling["type"] == PhaseType.CONCURRENCY
    assert profiling["requests"] == 10
    assert "sessions" not in profiling


@pytest.mark.parametrize("format_name", _DIRECT_GRAPH_FORMATS)
def test_explicit_graph_stop_condition_is_preserved_without_opening_payload(
    tmp_path: Path, format_name: str
) -> None:
    graph = _opaque_graph_file(tmp_path, format_name)
    cli = _make_cli(
        input_file=str(graph),
        custom_dataset_type=format_name,
        conversation_num=7,
    )

    with patch("builtins.open", side_effect=AssertionError("graph payload opened")):
        profiling = build_profiling(cli)

    assert profiling["sessions"] == 7
    assert "requests" not in profiling


@pytest.mark.parametrize("format_name", _DIRECT_GRAPH_FORMATS)
def test_fixed_schedule_graph_does_not_count_rows_or_probe_timestamps(
    tmp_path: Path, format_name: str
) -> None:
    graph = _opaque_graph_file(tmp_path, format_name)
    cli = _make_cli(
        input_file=str(graph),
        custom_dataset_type=format_name,
        fixed_schedule=True,
    )

    with patch("builtins.open", side_effect=AssertionError("graph payload opened")):
        profiling = build_profiling(cli)

    assert profiling["type"] == PhaseType.FIXED_SCHEDULE
    assert "requests" not in profiling
    assert "sessions" not in profiling


def test_python_owned_linear_dataset_keeps_ordinary_default(tmp_path: Path) -> None:
    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"prompt": "hi"}\n{"prompt": "yo"}\n')
    cli = _make_cli(input_file=str(plain), custom_dataset_type="single_turn")

    profiling = build_profiling(cli)

    assert profiling["requests"] == 10
    assert "sessions" not in profiling
