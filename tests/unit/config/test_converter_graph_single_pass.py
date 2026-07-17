# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bare graph run = single corpus pass (no auto-``--request-count 10`` truncation).

A bare ``aiperf profile --graph X.json`` with none of ``--request-count`` /
``--num-conversations`` / ``--benchmark-duration`` must NOT get the plain-aiperf
auto-10 stop condition (which truncates the benchmark to 10 node dispatches).
Instead the profiling ``ConcurrencyPhase`` stays UNBOUNDED and validates -- the
graph strategy then does a single corpus pass (each loaded trace once) driven by
``_recycle_has_stop_condition() is False``. Non-graph bare runs keep the auto-10.

The seam:
* ``_converter_profiling.build_profiling`` detects a graph workload from the CLI
  (``--graph-format`` set, or ``--input-file`` sniffed as a graph adapter) and,
  for a bare run, leaves the phase UNBOUNDED (no requests/duration/sessions)
  instead of injecting ``requests=10``.
* ``check_phase_dataset_compatibility`` (run from ``BenchmarkConfig``) accepts a
  no-stop concurrency phase ONLY when its dataset is a graph workload (mirrors
  how ``FixedSchedulePhase`` infers its stop from the dataset). A no-stop phase
  against a non-graph dataset is rejected there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aiperf.config.config import BenchmarkConfig
from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.phases import ConcurrencyPhase
from aiperf.plugin.enums import PhaseType

_WEKA_MIN = Path(__file__).parents[1] / "graph" / "fixtures" / "weka_min.json"


def _config_with(datasets: list[dict], phases: list[dict]) -> dict:
    """Minimal valid BenchmarkConfig dict for the given datasets + phases."""
    return {
        "models": ["test-model"],
        "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
        "datasets": datasets,
        "phases": phases,
    }


def _validate_profiling_phase(prof: dict) -> ConcurrencyPhase:
    """Validate the profiling prof-dict as the real ConcurrencyPhase.

    Mirrors ``converter._assemble_envelope_dict`` which prepends
    ``name="profiling"`` before handing the dict to the phase model.
    """
    return ConcurrencyPhase(name="profiling", **prof)


# ---------------------------------------------------------------------------
# Bare graph run: NO auto-10, phase unbounded but valid (single corpus pass)
# ---------------------------------------------------------------------------


def test_bare_graph_run_autodetected_has_no_stop_condition() -> None:
    """Bare ``--input-file <weka>`` (autodetected graph) -> no auto-10 bound."""
    cli = CLIConfig(model_names=["test-model"], input_file=str(_WEKA_MIN))
    prof = build_profiling(cli)

    assert prof["type"] == PhaseType.CONCURRENCY
    assert prof.get("requests") is None
    assert prof.get("duration") is None
    assert prof.get("sessions") is None
    # The stop_condition_inferred flag is gone; graph-ness now drives validity.
    assert "stop_condition_inferred" not in prof


def test_bare_graph_run_autodetected_phase_validates_unbounded() -> None:
    """The resulting unbounded concurrency phase VALIDATES at phase level."""
    cli = CLIConfig(model_names=["test-model"], input_file=str(_WEKA_MIN))
    phase = _validate_profiling_phase(build_profiling(cli))

    assert phase.requests is None
    assert phase.duration is None
    assert phase.sessions is None


def test_bare_graph_run_via_graph_format_flag_has_no_stop_condition() -> None:
    """Explicit ``--graph-format`` forces graph mode -> no auto-10 bound."""
    cli = CLIConfig(
        model_names=["test-model"],
        input_file=str(_WEKA_MIN),
        graph_format="weka_trace",
    )
    prof = build_profiling(cli)

    assert prof.get("requests") is None
    assert "stop_condition_inferred" not in prof
    _validate_profiling_phase(prof)  # must not raise


# ---------------------------------------------------------------------------
# Regression: NON-graph bare run still gets the auto-10 (unchanged behavior)
# ---------------------------------------------------------------------------


def test_non_graph_bare_run_still_gets_auto_10() -> None:
    """A synthetic (non-graph) bare run keeps the plain-aiperf ``requests=10``."""
    cli = CLIConfig(model_names=["test-model"])
    prof = build_profiling(cli)

    assert prof["type"] == PhaseType.CONCURRENCY
    assert prof["requests"] == 10
    assert "stop_condition_inferred" not in prof
    # And it validates with the auto-10 bound.
    assert _validate_profiling_phase(prof).requests == 10


def test_non_graph_file_bare_run_still_gets_auto_10(tmp_path: Path) -> None:
    """A plain (non-graph) single-turn file bare run keeps the auto-10."""
    input_file = tmp_path / "plain.jsonl"
    input_file.write_text('{"text": "hi"}\n')
    cli = CLIConfig(model_names=["test-model"], input_file=str(input_file))
    prof = build_profiling(cli)

    assert prof["requests"] == 10
    assert "stop_condition_inferred" not in prof


# ---------------------------------------------------------------------------
# Bounded graph run: an explicit bound is preserved (no inferred flag)
# ---------------------------------------------------------------------------


def test_bounded_graph_run_preserves_request_count() -> None:
    """``--request-count 25`` on a graph run keeps requests==25 (no inferred flag)."""
    cli = CLIConfig(
        model_names=["test-model"],
        input_file=str(_WEKA_MIN),
        request_count=25,
    )
    prof = build_profiling(cli)

    assert prof["requests"] == 25
    assert "stop_condition_inferred" not in prof


def test_bounded_graph_run_num_conversations_preserved() -> None:
    """``--num-conversations 7`` on a graph run keeps sessions==7 (no inferred flag)."""
    cli = CLIConfig(
        model_names=["test-model"],
        input_file=str(_WEKA_MIN),
        conversation_num=7,
    )
    prof = build_profiling(cli)

    assert prof["sessions"] == 7
    assert "stop_condition_inferred" not in prof


# ---------------------------------------------------------------------------
# Phase-model seam: the required-stop check no longer lives on the phase
# ---------------------------------------------------------------------------


def test_stop_condition_inferred_field_is_gone() -> None:
    """The user-facing opt-out flag was deleted entirely (no bypass foot-gun)."""
    assert "stop_condition_inferred" not in ConcurrencyPhase.model_fields


def test_bare_concurrency_phase_validates_at_phase_level() -> None:
    """A no-stop concurrency phase no longer raises at phase construction.

    The required-stop check moved to ``check_phase_dataset_compatibility`` (run
    from ``BenchmarkConfig``), which needs the dataset to decide -- so the phase
    model alone accepts a no-stop concurrency phase.
    """
    phase = ConcurrencyPhase(name="profiling", type=PhaseType.CONCURRENCY)
    assert phase.requests is None
    assert phase.duration is None
    assert phase.sessions is None


# ---------------------------------------------------------------------------
# BenchmarkConfig validation seam: graph exempts no-stop, non-graph rejects it
# ---------------------------------------------------------------------------


def test_graph_config_no_stop_condition_validates_via_graph_format() -> None:
    """A graph dataset (forced via ``graph_format``) + no-stop phase VALIDATES."""
    cfg = BenchmarkConfig(
        **_config_with(
            datasets=[
                {
                    "name": "main",
                    "type": "file",
                    "path": str(_WEKA_MIN),
                    "graph_format": "weka_trace",
                }
            ],
            phases=[{"name": "profiling", "type": "concurrency", "concurrency": 8}],
        )
    )
    profiling = next(p for p in cfg.phases if p.name == "profiling")
    assert profiling.requests is None
    assert profiling.duration is None
    assert profiling.sessions is None


def test_graph_config_no_stop_condition_validates_via_autodetect() -> None:
    """A graph dataset (auto-detected from the weka path) + no-stop phase VALIDATES."""
    cfg = BenchmarkConfig(
        **_config_with(
            datasets=[{"name": "main", "type": "file", "path": str(_WEKA_MIN)}],
            phases=[{"name": "profiling", "type": "concurrency", "concurrency": 8}],
        )
    )
    profiling = next(p for p in cfg.phases if p.name == "profiling")
    assert profiling.requests is None


def test_non_graph_config_no_stop_condition_raises() -> None:
    """A non-graph (synthetic) dataset + no-stop concurrency phase RAISES."""
    with pytest.raises(ValidationError, match="at least one of"):
        BenchmarkConfig(
            **_config_with(
                datasets=[
                    {
                        "name": "main",
                        "type": "synthetic",
                        "entries": 100,
                        "prompts": {"isl": 128, "osl": 64},
                    }
                ],
                phases=[{"name": "profiling", "type": "concurrency", "concurrency": 8}],
            )
        )
