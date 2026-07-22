# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MultiRunOrchestrator API and scenario-submission metadata tests."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from pytest import param

from aiperf.config.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkPlan
from aiperf.orchestrator.aggregation.base import AggregateResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator

_WEKA_LOADER = "semianalysis_cc_traces_weka_with_subagents"


# ---------------------------------------------------------------------------
# API surface guard
# ---------------------------------------------------------------------------


def test_obsolete_multi_run_orchestrator_api_is_absent() -> None:
    """The orchestrator accepts a plan/executor and exposes no legacy driver API."""
    sig = inspect.signature(MultiRunOrchestrator.__init__)
    params = sig.parameters

    assert "service_config" not in params
    assert "strategy" not in params
    assert "cell_callback" in params
    assert params["cell_callback"].kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        MultiRunOrchestrator(object(), object())  # type: ignore[call-arg]

    for obsolete_method in (
        "_execute_single_run",
        "_extract_summary_metrics",
        "_extract_was_cancelled",
        "_stamp_scenario_submission_metadata",
        "execute_and_export",
        "_resolve_strategy",
        "_execute_loop",
        "_create_sweep_strategy",
        "_create_confidence_strategy",
    ):
        assert not hasattr(MultiRunOrchestrator, obsolete_method), (
            f"obsolete orchestrator method {obsolete_method!r} unexpectedly exists"
        )


# ---------------------------------------------------------------------------
# Scenario-submission stamping
# ---------------------------------------------------------------------------


def _make_plan(
    *,
    scenario: str | None,
    streaming: bool | None = True,
    extra: dict[str, Any] | None = None,
    duration: Any = 1800,
    unsafe_override: bool = False,
) -> BenchmarkPlan:
    """Build a REAL BenchmarkPlan whose configs[0] carries a weka scenario.

    Mirrors the clean-weka-public-dataset shape from
    tests/unit/common/scenario/test_scenario_validator.py so ``apply_scenario``
    (invoked inside ``_stamp_scenario_submission_metadata``) produces
    ``submission_valid=True`` for the happy path. ``unsafe_override`` + explicit
    ``streaming=False`` yields a hard violation downgraded to a warning
    (``submission_valid=False``, reasons ``["unsafe_override"]``).
    """
    if extra is None:
        extra = {"ignore_eos": True}
    endpoint: dict[str, Any] = {
        "urls": ["http://localhost:8000/v1/chat/completions"],
        "type": "chat",
    }
    if streaming is not None:
        endpoint["streaming"] = streaming
    endpoint["extra"] = extra

    body: dict[str, Any] = {
        "models": ["my-model"],
        "endpoint": endpoint,
        "datasets": [{"name": "main", "type": "public", "dataset": _WEKA_LOADER}],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 8,
                "duration": duration,
            }
        ],
    }
    if scenario is not None:
        body["scenario"] = scenario
    if unsafe_override:
        body["unsafe_override"] = True

    cfg = BenchmarkConfig.model_validate(body)
    return BenchmarkPlan(configs=[cfg], trials=1)


def _make_aggregate() -> AggregateResult:
    return AggregateResult(
        aggregation_type="confidence",
        num_runs=0,
        num_successful_runs=0,
        failed_runs=[],
        metrics={},
        metadata={},
    )


def test_stamp_scenario_submission_metadata_clean_weka_marks_valid() -> None:
    """A clean weka scenario stamps name + submission_valid True + empty reasons.

    Builds a real BenchmarkPlan so ``apply_scenario`` re-resolution runs
    against the actual scenario spec.
    """
    from aiperf.cli_runner._aggregate import _stamp_scenario_submission_metadata

    plan = _make_plan(scenario="inferencex-agentx-mvp")
    aggregate = _make_aggregate()

    _stamp_scenario_submission_metadata(aggregate, [], plan)

    assert aggregate.metadata["_scenario_name"] == "inferencex-agentx-mvp"
    assert aggregate.metadata["_validator_submission_valid"] is True
    assert aggregate.metadata["_validator_submission_invalid_reasons"] == []
    assert aggregate.metadata["_total_responses"] == 0
    assert aggregate.metadata["_context_overflow_count"] == 0


def test_stamp_scenario_submission_metadata_no_scenario_is_noop() -> None:
    """No ``--scenario`` -> no scenario carrier keys (dataset provenance is
    still stamped for public datasets, independent of scenario)."""
    from aiperf.cli_runner._aggregate import _stamp_scenario_submission_metadata

    plan = _make_plan(scenario=None)
    aggregate = _make_aggregate()
    aggregate.metadata["pre_existing"] = "kept"

    _stamp_scenario_submission_metadata(aggregate, [], plan)

    assert "_scenario_name" not in aggregate.metadata
    assert "_validator_submission_valid" not in aggregate.metadata
    assert aggregate.metadata["pre_existing"] == "kept"
    # Public-dataset provenance is stamped even without a scenario.
    assert aggregate.metadata["dataset"]["source_type"] == "public_dataset"


def test_stamp_metadata_includes_public_dataset_provenance() -> None:
    """Multi-run aggregates retain the public dataset identity."""
    from aiperf.cli_runner._aggregate import _stamp_scenario_submission_metadata

    plan = _make_plan(scenario=None)
    dataset = plan.configs[0].get_default_dataset()
    dataset.entries = 393
    # ``entries_explicit`` is the converter's "user named --num-dataset-entries"
    # signal; only then does provenance emit num_dataset_entries.
    dataset.entries_explicit = True
    aggregate = _make_aggregate()

    _stamp_scenario_submission_metadata(aggregate, [], plan)

    assert aggregate.metadata["dataset"] == {
        "source_type": "public_dataset",
        "loader": _WEKA_LOADER,
        "hf_dataset_name": "semianalysisai/cc-traces-weka-062126",
        "hf_split": "train",
        "num_dataset_entries": 393,
    }


def test_stamp_metadata_omits_num_dataset_entries_when_not_explicit() -> None:
    """``entries`` derived from --num-conversations / --request-count (not the
    explicit --num-dataset-entries flag) must NOT surface num_dataset_entries.

    Mirrors cquil's gate on ``num_dataset_entries in model_fields_set`` for the
    distinct ConversationConfig field that --request-count / --num-conversations
    never populate. In v2 those flags DO populate ``entries`` (it is the live
    entry-limit), so provenance keys off ``entries_explicit`` instead.
    """
    from aiperf.cli_runner._aggregate import _stamp_scenario_submission_metadata

    plan = _make_plan(scenario=None)
    dataset = plan.configs[0].get_default_dataset()
    # Converter-derived fallback: entries is set, but the user never named
    # --num-dataset-entries, so entries_explicit stays False.
    dataset.entries = 500
    assert dataset.entries_explicit is False
    aggregate = _make_aggregate()

    _stamp_scenario_submission_metadata(aggregate, [], plan)

    assert aggregate.metadata["dataset"] == {
        "source_type": "public_dataset",
        "loader": _WEKA_LOADER,
        "hf_dataset_name": "semianalysisai/cc-traces-weka-062126",
        "hf_split": "train",
    }
    assert "num_dataset_entries" not in aggregate.metadata["dataset"]


def test_stamp_scenario_submission_metadata_unsafe_override_marks_invalid() -> None:
    """An unsafe-override violation stamps submission_valid False + reasons.

    Explicit ``--streaming=false`` against a require-streaming scenario is a hard
    violation; ``unsafe_override`` downgrades it to a warning and
    ``apply_scenario`` returns ``submission_valid=False`` with the
    ``unsafe_override`` reason tag.
    """
    from aiperf.cli_runner._aggregate import _stamp_scenario_submission_metadata

    plan = _make_plan(
        scenario="inferencex-agentx-mvp",
        streaming=False,
        unsafe_override=True,
    )
    aggregate = _make_aggregate()

    _stamp_scenario_submission_metadata(aggregate, [], plan)

    assert aggregate.metadata["_scenario_name"] == "inferencex-agentx-mvp"
    assert aggregate.metadata["_validator_submission_valid"] is False
    assert (
        "unsafe_override"
        in (aggregate.metadata["_validator_submission_invalid_reasons"])
    )


@pytest.mark.parametrize(
    ("per_run_cancelled", "expected"),
    [
        param([False, False], False, id="none-cancelled"),
        param([False, True], True, id="one-cancelled"),
    ],
)  # fmt: skip
def test_stamp_scenario_submission_metadata_carries_was_cancelled(
    per_run_cancelled: list[bool], expected: bool
) -> None:
    """Any cancelled run in the batch flips the ``_was_cancelled`` carrier key.

    The per-run flag rides on ``RunResult.was_cancelled`` (populated by
    LocalSubprocessExecutor from the run's profile export); the aggregate stamp
    is ``any(r.was_cancelled for r in results)`` and is consumed by
    AggregateConfidenceJsonExporter to mark ``submission_valid=False`` with
    reason ``run_cancelled``.
    """
    from aiperf.cli_runner._aggregate import _stamp_scenario_submission_metadata
    from aiperf.orchestrator.models import RunResult

    plan = _make_plan(scenario="inferencex-agentx-mvp")
    results = [
        RunResult(label=f"run_{i:04d}", success=True, was_cancelled=cancelled)
        for i, cancelled in enumerate(per_run_cancelled)
    ]
    aggregate = _make_aggregate()

    _stamp_scenario_submission_metadata(aggregate, results, plan)

    assert aggregate.metadata["_was_cancelled"] is expected
