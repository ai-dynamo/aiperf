# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writer side of the multi-run AGGREGATE scenario-submission verdict.

The reader (``AggregateConfidenceJsonExporter._build_submission_metadata``) was
ported but the WRITER was not, so multi-run aggregate exports emitted an EMPTY
submission verdict (no ``scenario`` / ``submission_valid`` keys). These tests
cover ``cli_runner._aggregate._stamp_scenario_submission_metadata``: it stamps
the carrier keys the reader pops, and the writer + reader round-trip produces
the right aggregate verdict (including cross-run overflow rate, cancellation,
and the ``--unsafe-override`` lock fold).

Uses real ``BenchmarkPlan`` / ``BenchmarkConfig`` / ``RunResult`` /
``AggregateResult`` objects (no MagicMock) so the full plumbing is exercised.
"""

from __future__ import annotations

import orjson

from aiperf.cli_runner._aggregate import (
    SCENARIO_SUBMISSION_CARRIER_KEYS,
    _stamp_scenario_submission_metadata,
    _sum_runtime_response_counts,
    strip_scenario_submission_carrier_keys,
)
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config import BenchmarkConfig, BenchmarkPlan
from aiperf.exporters.aggregate import (
    AggregateConfidenceCsvExporter,
    AggregateConfidenceJsonExporter,
    AggregateExporterConfig,
)
from aiperf.orchestrator.aggregation.base import AggregateResult
from aiperf.orchestrator.models import RunResult

_MINIMAL_CONFIG = {
    "models": ["test-model"],
    "endpoint": {
        "urls": ["http://localhost:8000/v1/chat/completions"],
        "wait_for_model_timeout": 0,
    },
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 100,
            "concurrency": 1,
        }
    ],
}


def _make_plan(*, scenario: str | None, unsafe_override: bool = False) -> BenchmarkPlan:
    """A non-sweep two-trial plan, optionally carrying a scenario lock."""
    cfg_dict = dict(_MINIMAL_CONFIG)
    if scenario is not None:
        cfg_dict = {
            **cfg_dict,
            "scenario": scenario,
            "unsafe_override": unsafe_override,
        }
    cfg = BenchmarkConfig.model_validate(cfg_dict)
    return BenchmarkPlan(configs=[cfg], trials=2, confidence_level=0.95)


def _count(value: float) -> JsonMetricResult:
    """A count-style ``JsonMetricResult`` whose per-run total is ``avg``."""
    return JsonMetricResult(unit="requests", avg=value, min=value, max=value)


def _run(
    *,
    label: str,
    artifacts_path,
    request_count: float,
    overflow: float = 0.0,
    errors: float = 0.0,
    was_cancelled: bool = False,
    success: bool = True,
    submission_valid: bool | None = True,
    submission_invalid_reasons: list[str] | None = None,
) -> RunResult:
    """A ``RunResult`` carrying the real lock-only verdict + a per-run JSON.

    ``submission_valid`` / ``submission_invalid_reasons`` mirror the
    ``ScenarioOutcome`` the orchestrator carries onto each run from
    ``run.resolved.scenario_outcome`` (defaults to a clean ``(True, [])`` lock).
    The per-run JSON records ``was_cancelled`` so the writer's cross-run
    cancellation fold (which reads it back) is exercised.
    """
    artifacts_path.mkdir(parents=True, exist_ok=True)
    (artifacts_path / "profile_export_aiperf.json").write_bytes(
        orjson.dumps({"was_cancelled": was_cancelled})
    )
    return RunResult(
        label=label,
        success=success,
        artifacts_path=artifacts_path,
        summary_metrics={
            "request_count": _count(request_count),
            "error_request_count": _count(errors),
            "context_overflow_count": _count(overflow),
        },
        submission_valid=submission_valid,
        submission_invalid_reasons=list(submission_invalid_reasons or []),
    )


def _aggregate(metadata: dict) -> AggregateResult:
    return AggregateResult(
        aggregation_type="confidence",
        num_runs=2,
        num_successful_runs=2,
        failed_runs=[],
        metrics={},
        metadata=dict(metadata),
    )


def _export_metadata(tmp_path, result: AggregateResult) -> dict:
    config = AggregateExporterConfig(result=result, output_dir=tmp_path)
    exporter = AggregateConfidenceJsonExporter(config=config)
    return orjson.loads(exporter._generate_content())["metadata"]


def test_sum_runtime_response_counts_sums_across_runs(tmp_path):
    """Cross-run sum = request + error + overflow, with overflow double-counted."""
    runs = [
        _run(
            label="r0",
            artifacts_path=tmp_path / "r0",
            request_count=90,
            overflow=2,
            errors=3,
        ),
        _run(
            label="r1",
            artifacts_path=tmp_path / "r1",
            request_count=80,
            overflow=4,
            errors=1,
        ),
    ]
    total, overflow = _sum_runtime_response_counts(runs)
    assert overflow == 6
    assert total == (90 + 3 + 2) + (80 + 1 + 4)


def test_writer_stamps_all_carrier_keys_clean_run(tmp_path):
    """A clean scenario run stamps a True validator verdict + summed counts."""
    plan = _make_plan(scenario="inferencex-agentx-mvp")
    aggregate = _aggregate({"confidence_level": 0.95})
    runs = [
        _run(label="r0", artifacts_path=tmp_path / "r0", request_count=100, overflow=1),
        _run(label="r1", artifacts_path=tmp_path / "r1", request_count=100, overflow=0),
    ]
    _stamp_scenario_submission_metadata(aggregate, runs, plan)
    md = aggregate.metadata
    assert md["_scenario_name"] == "inferencex-agentx-mvp"
    assert md["_validator_submission_valid"] is True
    assert md["_validator_submission_invalid_reasons"] == []
    assert md["_total_responses"] == 201
    assert md["_context_overflow_count"] == 1
    assert md["_was_cancelled"] is False


def test_writer_reader_roundtrip_clean_run_valid(tmp_path):
    """Writer + reader round-trip: a clean 2-run scenario aggregate is valid."""
    plan = _make_plan(scenario="inferencex-agentx-mvp")
    aggregate = _aggregate({"confidence_level": 0.95})
    runs = [
        _run(
            label="r0", artifacts_path=tmp_path / "r0", request_count=1000, overflow=1
        ),
        _run(
            label="r1", artifacts_path=tmp_path / "r1", request_count=1000, overflow=0
        ),
    ]
    _stamp_scenario_submission_metadata(aggregate, runs, plan)
    md = _export_metadata(tmp_path / "agg", aggregate)
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert md["submission_valid"] is True
    assert "submission_invalid_reasons" not in md


def test_writer_reader_roundtrip_unsafe_override_violation_invalid(tmp_path):
    """``--unsafe-override`` WITH a real violation folds invalid across runs.

    The invalid verdict comes from the lock-only ``ScenarioOutcome`` carried on
    each run (``submission_valid=False``, reasons ``['unsafe_override']``) -- the
    real outcome ``apply_scenario`` produces when violations are downgraded -- NOT
    from re-deriving the verdict from the ``unsafe_override`` flag.
    """
    plan = _make_plan(scenario="inferencex-agentx-mvp", unsafe_override=True)
    aggregate = _aggregate({"confidence_level": 0.95})
    runs = [
        _run(
            label="r0",
            artifacts_path=tmp_path / "r0",
            request_count=100,
            submission_valid=False,
            submission_invalid_reasons=["unsafe_override"],
        ),
        _run(
            label="r1",
            artifacts_path=tmp_path / "r1",
            request_count=100,
            submission_valid=False,
            submission_invalid_reasons=["unsafe_override"],
        ),
    ]
    _stamp_scenario_submission_metadata(aggregate, runs, plan)
    md = _export_metadata(tmp_path / "agg", aggregate)
    assert md["submission_valid"] is False
    assert md["submission_invalid_reasons"] == ["unsafe_override"]


def test_writer_reader_roundtrip_clean_override_no_violation_valid(tmp_path):
    """C-1 regression: a CLEAN lock carrying ``--unsafe-override`` is VALID.

    ``apply_scenario`` returns ``(True, [])`` for a conforming config even with
    ``--unsafe-override`` set (the flag only matters when there are violations to
    downgrade). The old writer re-derived the verdict from the flag alone and
    falsely stamped ``submission_valid=False`` / ``['unsafe_override']``. With the
    real carried outcome on each run, the aggregate matches the single-run path.
    """
    plan = _make_plan(scenario="inferencex-agentx-mvp", unsafe_override=True)
    aggregate = _aggregate({"confidence_level": 0.95})
    runs = [
        _run(
            label="r0",
            artifacts_path=tmp_path / "r0",
            request_count=100,
            submission_valid=True,
            submission_invalid_reasons=[],
        ),
        _run(
            label="r1",
            artifacts_path=tmp_path / "r1",
            request_count=100,
            submission_valid=True,
            submission_invalid_reasons=[],
        ),
    ]
    _stamp_scenario_submission_metadata(aggregate, runs, plan)
    assert aggregate.metadata["_validator_submission_valid"] is True
    assert aggregate.metadata["_validator_submission_invalid_reasons"] == []
    md = _export_metadata(tmp_path / "agg", aggregate)
    assert md["submission_valid"] is True
    assert "submission_invalid_reasons" not in md


def test_writer_folds_cancellation_over_failed_run(tmp_path):
    """I-1 regression: cancellation folds over ALL runs, not just successful ones.

    A graceful Ctrl+C that completed ZERO requests is classified
    ``success=False`` but still wrote its per-run JSON with ``was_cancelled=true``.
    Folding only over successful runs would drop it; folding over all runs (as
    AgentX does) keeps the aggregate invalid.
    """
    plan = _make_plan(scenario="inferencex-agentx-mvp")
    aggregate = _aggregate({"confidence_level": 0.95})
    runs = [
        _run(label="r0", artifacts_path=tmp_path / "r0", request_count=100),
        _run(
            label="r1",
            artifacts_path=tmp_path / "r1",
            request_count=0,
            success=False,
            was_cancelled=True,
        ),
    ]
    _stamp_scenario_submission_metadata(aggregate, runs, plan)
    assert aggregate.metadata["_was_cancelled"] is True
    md = _export_metadata(tmp_path / "agg", aggregate)
    assert md["submission_valid"] is False
    assert "run_cancelled" in md["submission_invalid_reasons"]


def test_writer_reader_roundtrip_cross_run_overflow_rate(tmp_path):
    """Cross-run overflow rate > 1% flips the aggregate invalid via summed counts.

    Neither run alone trips it cleanly, but the summed rate
    (6 / 200 = 3% > 1%) does, proving the cross-run fold.
    """
    plan = _make_plan(scenario="inferencex-agentx-mvp")
    aggregate = _aggregate({"confidence_level": 0.95})
    runs = [
        _run(label="r0", artifacts_path=tmp_path / "r0", request_count=97, overflow=3),
        _run(label="r1", artifacts_path=tmp_path / "r1", request_count=97, overflow=3),
    ]
    _stamp_scenario_submission_metadata(aggregate, runs, plan)
    assert aggregate.metadata["_total_responses"] == 200
    assert aggregate.metadata["_context_overflow_count"] == 6
    md = _export_metadata(tmp_path / "agg", aggregate)
    assert md["submission_valid"] is False
    assert "context_overflow_rate_exceeded" in md["submission_invalid_reasons"]


def test_writer_reader_roundtrip_cancellation_invalid(tmp_path):
    """A cancelled run (read from per-run JSON) flips the aggregate invalid."""
    plan = _make_plan(scenario="inferencex-agentx-mvp")
    aggregate = _aggregate({"confidence_level": 0.95})
    runs = [
        _run(label="r0", artifacts_path=tmp_path / "r0", request_count=50),
        _run(
            label="r1",
            artifacts_path=tmp_path / "r1",
            request_count=50,
            was_cancelled=True,
        ),
    ]
    _stamp_scenario_submission_metadata(aggregate, runs, plan)
    assert aggregate.metadata["_was_cancelled"] is True
    md = _export_metadata(tmp_path / "agg", aggregate)
    assert md["submission_valid"] is False
    assert md["submission_invalid_reasons"] == ["run_cancelled"]


def test_writer_no_scenario_is_noop(tmp_path):
    """No scenario -> no carrier keys stamped -> reader omits submission fields."""
    plan = _make_plan(scenario=None)
    aggregate = _aggregate({"confidence_level": 0.95})
    runs = [
        _run(label="r0", artifacts_path=tmp_path / "r0", request_count=10, overflow=9),
        _run(label="r1", artifacts_path=tmp_path / "r1", request_count=10, overflow=9),
    ]
    _stamp_scenario_submission_metadata(aggregate, runs, plan)
    for key in (
        "_scenario_name",
        "_validator_submission_valid",
        "_total_responses",
        "_was_cancelled",
    ):
        assert key not in aggregate.metadata
    md = _export_metadata(tmp_path / "agg", aggregate)
    assert "scenario" not in md
    assert "submission_valid" not in md


def test_writer_stamps_exactly_the_carrier_key_constant(tmp_path):
    """Drift guard: the stamped key set IS ``SCENARIO_SUBMISSION_CARRIER_KEYS``.

    The CSV exporter strips exactly this constant; a new carrier key stamped
    without extending it would leak into user-facing output.
    """
    plan = _make_plan(scenario="inferencex-agentx-mvp")
    aggregate = _aggregate({})
    runs = [_run(label="r0", artifacts_path=tmp_path / "r0", request_count=10)]
    _stamp_scenario_submission_metadata(aggregate, runs, plan)
    assert set(aggregate.metadata) == set(SCENARIO_SUBMISSION_CARRIER_KEYS)


def test_strip_helper_removes_only_carrier_keys():
    """The shared helper strips every carrier key and nothing else."""
    metadata = {key: "x" for key in SCENARIO_SUBMISSION_CARRIER_KEYS}
    metadata["cooldown_seconds"] = 5
    metadata["scenario"] = "kept"
    stripped = strip_scenario_submission_carrier_keys(metadata)
    assert stripped == {"cooldown_seconds": 5, "scenario": "kept"}
    # Input not mutated.
    assert set(SCENARIO_SUBMISSION_CARRIER_KEYS) <= set(metadata)


def test_csv_exporter_excludes_carrier_keys(tmp_path):
    """WK3: the aggregate confidence CSV must not leak the carrier keys.

    The CSV title-cases metadata keys (``_scenario_name`` -> `` Scenario Name``),
    so assert on the rendered names as well as the raw keys.
    """
    plan = _make_plan(scenario="inferencex-agentx-mvp")
    aggregate = _aggregate({"confidence_level": 0.95, "cooldown_seconds": 3})
    runs = [
        _run(label="r0", artifacts_path=tmp_path / "r0", request_count=100, overflow=1),
        _run(label="r1", artifacts_path=tmp_path / "r1", request_count=100, overflow=0),
    ]
    _stamp_scenario_submission_metadata(aggregate, runs, plan)

    exporter = AggregateConfidenceCsvExporter(
        config=AggregateExporterConfig(result=aggregate, output_dir=tmp_path / "agg")
    )
    content = exporter._generate_content()

    for key in SCENARIO_SUBMISSION_CARRIER_KEYS:
        assert key not in content
        assert key.replace("_", " ").title() not in content
    # Non-carrier metadata still exported.
    assert "Cooldown Seconds" in content
    assert "Confidence Level" in content
