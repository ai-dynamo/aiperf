# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A bare graph run stays unbounded for a single corpus pass; non-graph runs keep the auto ``--request-count 10``."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.config import BenchmarkConfig
from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.phases import ConcurrencyPhase
from aiperf.plugin.enums import PhaseType
from tests.unit.config.conftest import GRAPH_TRACE_FIXTURE as _GRAPH_MIN


def _config_with(datasets: list[dict], phases: list[dict]) -> dict:
    """Minimal valid BenchmarkConfig dict for the given datasets + phases."""
    return {
        "models": ["test-model"],
        "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
        "datasets": datasets,
        "phases": phases,
    }


def _validate_profiling_phase(prof: dict) -> ConcurrencyPhase:
    """Validate a prof-dict as a real ConcurrencyPhase, mirroring the converter's ``name="profiling"`` prepend."""
    return ConcurrencyPhase(name="profiling", **prof)


def _graph_cli(**overrides: object) -> CLIConfig:
    """CLIConfig whose ``--input-file`` is the real dynamo graph trace fixture."""
    return CLIConfig(
        model_names=["test-model"], input_file=str(_GRAPH_MIN), **overrides
    )


@pytest.mark.parametrize(
    "overrides",
    [
        param({}, id="autodetected-from-input-file"),
        param(
            {"graph_format": "dynamo_trace"}, id="forced-via-graph-format-flag"
        ),
    ],
)  # fmt: skip
def test_bare_graph_run_has_no_stop_condition(overrides: dict) -> None:
    """A bare graph run gets no auto-10 bound and no ``stop_condition_inferred`` marker."""
    prof = build_profiling(_graph_cli(**overrides))

    assert prof["type"] == PhaseType.CONCURRENCY
    assert prof.get("requests") is None
    assert prof.get("duration") is None
    assert prof.get("sessions") is None
    # The stop_condition_inferred flag is gone; graph-ness now drives validity.
    assert "stop_condition_inferred" not in prof
    _validate_profiling_phase(prof)  # must not raise


def test_bare_graph_run_phase_validates_unbounded() -> None:
    """The unbounded profiling phase from a bare graph run validates at phase level."""
    phase = _validate_profiling_phase(build_profiling(_graph_cli()))

    assert phase.requests is None
    assert phase.duration is None
    assert phase.sessions is None


def test_non_graph_bare_run_still_gets_auto_10() -> None:
    """A synthetic bare run keeps the plain-aiperf ``requests=10`` and validates with it."""
    prof = build_profiling(CLIConfig(model_names=["test-model"]))

    assert prof["type"] == PhaseType.CONCURRENCY
    assert prof["requests"] == 10
    assert "stop_condition_inferred" not in prof
    assert _validate_profiling_phase(prof).requests == 10


def test_non_graph_file_bare_run_still_gets_auto_10(tmp_path: Path) -> None:
    """A plain single-turn file bare run keeps the auto-10."""
    input_file = tmp_path / "plain.jsonl"
    input_file.write_text('{"text": "hi"}\n')
    prof = build_profiling(
        CLIConfig(model_names=["test-model"], input_file=str(input_file))
    )

    assert prof["requests"] == 10
    assert "stop_condition_inferred" not in prof


@pytest.mark.parametrize(
    ("overrides", "field", "expected"),
    [
        param({"request_count": 25}, "requests", 25, id="request-count-25"),
        param({"conversation_num": 7}, "sessions", 7, id="num-conversations-7"),
    ],
)  # fmt: skip
def test_bounded_graph_run_preserves_explicit_bound(
    overrides: dict, field: str, expected: int
) -> None:
    """An explicit bound on a graph run is preserved verbatim with no inferred flag."""
    prof = build_profiling(_graph_cli(**overrides))

    assert prof[field] == expected
    assert "stop_condition_inferred" not in prof


def test_stop_condition_inferred_field_is_gone() -> None:
    """The user-facing opt-out flag was deleted entirely, leaving no bypass foot-gun."""
    assert "stop_condition_inferred" not in ConcurrencyPhase.model_fields


def test_bare_concurrency_phase_validates_at_phase_level() -> None:
    """A no-stop concurrency phase constructs fine; the required-stop check moved to dataset-aware BenchmarkConfig validation."""
    phase = ConcurrencyPhase(name="profiling", type=PhaseType.CONCURRENCY)
    assert phase.requests is None
    assert phase.duration is None
    assert phase.sessions is None


@pytest.mark.parametrize(
    "dataset",
    [
        param(
            {
                "name": "main",
                "type": "file",
                "path": str(_GRAPH_MIN),
                "graph_format": "dynamo_trace",
            },
            id="graph-forced-via-graph-format",
        ),
        param(
            {"name": "main", "type": "file", "path": str(_GRAPH_MIN)},
            id="graph-autodetected-from-path",
        ),
    ],
)  # fmt: skip
def test_graph_config_no_stop_condition_validates(dataset: dict) -> None:
    """A graph dataset plus a no-stop concurrency phase validates at BenchmarkConfig level."""
    cfg = BenchmarkConfig(
        **_config_with(
            datasets=[dataset],
            phases=[{"name": "profiling", "type": "concurrency", "concurrency": 8}],
        )
    )
    profiling = next(p for p in cfg.phases if p.name == "profiling")
    assert profiling.requests is None
    assert profiling.duration is None
    assert profiling.sessions is None


def test_non_graph_config_no_stop_condition_raises() -> None:
    """A synthetic dataset plus a no-stop concurrency phase is rejected."""
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
