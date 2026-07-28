# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for speculative-decoding acceptance metrics.

Covers per-request value extraction, clean degradation when the neutral record
is absent, metric metadata (flags / console group / unit), a worked example
against a known per-step accepted-draft sequence, the run-level derived values,
the phase-scoped pooled histogram (sum + reconciliation with
``total_spec_decode_steps`` + warmup exclusion), and the one-line console
histogram cap/overflow rendering.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import (
    CreditPhase,
    GenericMetricUnit,
    MetricConsoleGroup,
    MetricFlags,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import MetricRecordMetadata, ParsedResponseRecord
from aiperf.common.models.spec_decode_models import SpecDecodeAcceptanceRecord
from aiperf.exporters.console_spec_decode_exporter import (
    SPEC_DECODE_HISTOGRAM_CONSOLE_CAP,
    format_acceptance_histogram_line,
)
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.spec_decode_metrics import (
    SpecDecodeAcceptanceLengthMetric,
    SpecDecodeAcceptedDraftTokensMetric,
    SpecDecodeAcceptedPerVerifiedMetric,
    SpecDecodeDraftAcceptanceRateMetric,
    SpecDecodeDraftTokensMetric,
    SpecDecodeOverallDraftAcceptanceRateMetric,
    SpecDecodeStepsMetric,
    SpecDecodeTokenWeightedAcceptanceLengthMetric,
    TotalAcceptedDraftTokensMetric,
    TotalDraftTokensMetric,
    TotalSpecDecodeStepsMetric,
)
from aiperf.records.records_manager import _pooled_spec_decode_histogram
from tests.unit.conftest import make_benchmark_run
from tests.unit.metrics.conftest import create_record

WORKED_EXAMPLE = [2, 3, 1, 4, 2, 0, 3, 3]
"""Ticket worked example: per-step accepted-draft counts. Histogram
{0:1, 1:1, 2:2, 3:3, 4:1}; 8 steps; 18 accepted; mean AL 1 + 18/8 = 3.25."""


def make_spec(
    per_step_accepted: list[int],
    k: int = 4,
    engine: str = "vllm",
) -> SpecDecodeAcceptanceRecord:
    """Build a neutral acceptance record from a per-step accepted-draft list.

    Drafts-per-step is the fixed block ``k``, so num_draft = k * steps. All
    aggregate fields are derived so the record satisfies its own invariants.
    """
    histogram = dict(Counter(per_step_accepted))
    steps = len(per_step_accepted)
    accepted = sum(per_step_accepted)
    draft = k * steps
    return SpecDecodeAcceptanceRecord(
        engine=engine,
        mean_acceptance_length=(1 + accepted / steps) if steps else 1.0,
        draft_acceptance_rate=(accepted / draft) if draft else 0.0,
        acceptance_histogram=histogram,
        num_accepted_draft_tokens=accepted,
        num_draft_tokens=draft,
        num_spec_steps=steps,
        num_spec_tokens=k,
    )


def spec_record(spec: SpecDecodeAcceptanceRecord | None) -> ParsedResponseRecord:
    """A valid single-response record carrying the given neutral record."""
    record = create_record(input_tokens=10)
    record.spec_decode_acceptance = spec
    return record


class TestWorkedExample:
    def test_histogram_matches_per_step_sequence(self):
        spec = make_spec(WORKED_EXAMPLE)
        assert spec.acceptance_histogram == {0: 1, 1: 1, 2: 2, 3: 3, 4: 1}
        assert spec.num_spec_steps == 8
        assert spec.num_accepted_draft_tokens == 18

    def test_mean_acceptance_length_is_3_25(self):
        record = spec_record(make_spec(WORKED_EXAMPLE))
        value = SpecDecodeAcceptanceLengthMetric().parse_record(
            record, MetricRecordDict()
        )
        assert value == pytest.approx(3.25)


class TestValueExtraction:
    @pytest.fixture
    def record(self) -> ParsedResponseRecord:
        return spec_record(make_spec(WORKED_EXAMPLE))

    @pytest.mark.parametrize(
        "metric_cls, expected",
        [
            param(SpecDecodeStepsMetric, 8, id="steps"),
            param(SpecDecodeAcceptedDraftTokensMetric, 18, id="accepted_draft"),
            param(SpecDecodeDraftTokensMetric, 32, id="draft"),
            param(SpecDecodeAcceptanceLengthMetric, 3.25, id="acceptance_length"),
            # draft_acceptance_rate is scaled to a percent: 18/32 * 100 = 56.25
            param(SpecDecodeDraftAcceptanceRateMetric, 56.25, id="draft_rate_pct"),
            # accepted-per-verified: (18 + 8) / (32 + 8) = 26 / 40 = 0.65
            param(SpecDecodeAcceptedPerVerifiedMetric, 0.65, id="accepted_per_verified"),
        ],
    )  # fmt: skip
    def test_extracts_expected_value(self, record, metric_cls, expected):
        value = metric_cls().parse_record(record, MetricRecordDict())
        assert value == pytest.approx(expected)

    def test_accepted_per_verified_undefined_raises(self):
        # No drafts and no verify steps -> the ratio denominator is zero.
        record = spec_record(make_spec([], k=0))
        with pytest.raises(NoMetricValue):
            SpecDecodeAcceptedPerVerifiedMetric().parse_record(
                record, MetricRecordDict()
            )


class TestDerivedZeroDenominatorGuards:
    """The run-level derived acceptance metrics raise ``NoMetricValue`` rather
    than dividing by zero when the run recorded no verify steps / no proposed
    draft tokens (the fully-rejected / zero-step edge). ``accepted_per_verified``
    has the analogous per-request guard covered in ``TestValueExtraction``.
    """

    def test_token_weighted_raises_on_zero_steps(self):
        results = MetricResultsDict()
        results[TotalSpecDecodeStepsMetric.tag] = 0
        results[TotalAcceptedDraftTokensMetric.tag] = 0
        with pytest.raises(NoMetricValue):
            SpecDecodeTokenWeightedAcceptanceLengthMetric().derive_value(results)

    def test_overall_draft_rate_raises_on_zero_draft(self):
        results = MetricResultsDict()
        results[TotalDraftTokensMetric.tag] = 0
        results[TotalAcceptedDraftTokensMetric.tag] = 0
        with pytest.raises(NoMetricValue):
            SpecDecodeOverallDraftAcceptanceRateMetric().derive_value(results)


class TestAbsentRecordDegradesCleanly:
    @pytest.mark.parametrize(
        "metric_cls",
        [
            param(SpecDecodeAcceptanceLengthMetric, id="acceptance_length"),
            param(SpecDecodeDraftAcceptanceRateMetric, id="draft_rate"),
            param(SpecDecodeAcceptedPerVerifiedMetric, id="accepted_per_verified"),
            param(SpecDecodeStepsMetric, id="steps"),
            param(SpecDecodeAcceptedDraftTokensMetric, id="accepted_draft"),
            param(SpecDecodeDraftTokensMetric, id="draft"),
        ],
    )  # fmt: skip
    def test_raises_no_metric_value_when_record_absent(self, metric_cls):
        record = spec_record(None)
        with pytest.raises(NoMetricValue):
            metric_cls().parse_record(record, MetricRecordDict())


class TestMetadata:
    @pytest.mark.parametrize(
        "metric_cls, unit, group, larger_is_better",
        [
            param(SpecDecodeAcceptanceLengthMetric, GenericMetricUnit.RATIO, MetricConsoleGroup.SPEC_DECODE, True, id="acceptance_length"),
            param(SpecDecodeDraftAcceptanceRateMetric, GenericMetricUnit.PERCENT, MetricConsoleGroup.SPEC_DECODE, True, id="draft_rate"),
            param(SpecDecodeAcceptedPerVerifiedMetric, GenericMetricUnit.RATIO, MetricConsoleGroup.SPEC_DECODE, True, id="accepted_per_verified"),
            param(SpecDecodeStepsMetric, GenericMetricUnit.COUNT, MetricConsoleGroup.SPEC_DECODE, False, id="steps"),
            param(SpecDecodeAcceptedDraftTokensMetric, GenericMetricUnit.TOKENS, MetricConsoleGroup.NONE, False, id="accepted_draft"),
            param(SpecDecodeDraftTokensMetric, GenericMetricUnit.TOKENS, MetricConsoleGroup.NONE, False, id="draft"),
            param(SpecDecodeTokenWeightedAcceptanceLengthMetric, GenericMetricUnit.RATIO, MetricConsoleGroup.SPEC_DECODE, True, id="token_weighted"),
            param(SpecDecodeOverallDraftAcceptanceRateMetric, GenericMetricUnit.PERCENT, MetricConsoleGroup.SPEC_DECODE, True, id="overall_rate"),
        ],
    )  # fmt: skip
    def test_metric_metadata(self, metric_cls, unit, group, larger_is_better):
        assert metric_cls.unit == unit
        assert metric_cls.console_group == group
        assert metric_cls.has_flags(MetricFlags.LARGER_IS_BETTER) == larger_is_better

    def test_no_hidden_flags(self):
        # EXPERIMENTAL/INTERNAL would hide from console AND files;
        # NO_INDIVIDUAL_RECORDS would drop per-record JSONL. None are allowed.
        forbidden = (
            MetricFlags.EXPERIMENTAL
            | MetricFlags.INTERNAL
            | MetricFlags.NO_INDIVIDUAL_RECORDS
        )
        for metric_cls in (
            SpecDecodeAcceptanceLengthMetric,
            SpecDecodeDraftAcceptanceRateMetric,
            SpecDecodeAcceptedPerVerifiedMetric,
            SpecDecodeStepsMetric,
        ):
            assert metric_cls.missing_flags(forbidden)


def _spec_metric_records_data(
    session_num: int,
    spec: SpecDecodeAcceptanceRecord | None,
    phase: CreditPhase = CreditPhase.PROFILING,
    phase_index: int | None = None,
) -> MetricRecordsData:
    """A wire record carrying the per-request spec-decode metrics + neutral struct."""
    metrics: dict = {}
    if spec is not None:
        metrics = {
            SpecDecodeStepsMetric.tag: spec.num_spec_steps,
            SpecDecodeAcceptedDraftTokensMetric.tag: spec.num_accepted_draft_tokens,
            SpecDecodeDraftTokensMetric.tag: spec.num_draft_tokens,
        }
    return MetricRecordsData(
        metadata=MetricRecordMetadata(
            session_num=session_num,
            request_start_ns=1000 + session_num,
            request_end_ns=2000 + session_num,
            conversation_id="conv",
            turn_index=session_num,
            record_processor_id="rp",
            benchmark_phase=phase,
            phase_index=phase_index,
            worker_id="worker",
        ),
        metrics=metrics,
        spec_decode_acceptance=spec,
        error=None,
    )


async def _summarize_specs(*specs: SpecDecodeAcceptanceRecord | None):
    """Feed each spec as a profiling record and return the accumulator summary."""
    acc = MetricsAccumulator(make_benchmark_run())
    for session_num, spec in enumerate(specs):
        await acc.process_record(_spec_metric_records_data(session_num, spec))
    return await acc.summarize()


# Two profiling records: WORKED_EXAMPLE (8 steps, 18 accepted, 32 draft) and
# [1,1,0] (3 steps, 2 accepted, 12 draft). Totals: 11 / 20 / 44.
_TWO_RECORD_SPECS = (WORKED_EXAMPLE, [1, 1, 0])


class TestPooledHistogram:
    def test_pool_sums_and_reconciles_with_total_steps(self):
        summary = asyncio.run(
            _summarize_specs(*(make_spec(s) for s in _TWO_RECORD_SPECS))
        )
        pooled = summary.pooled_spec_decode_acceptance_histogram
        # Elementwise pool of {0:1,1:1,2:2,3:3,4:1} and {0:1,1:2}.
        assert pooled == {0: 2, 1: 3, 2: 2, 3: 3, 4: 1}
        # Reconciliation: bucket counts sum to total_spec_decode_steps == 11.
        assert sum(pooled.values()) == 11
        assert summary.results[TotalSpecDecodeStepsMetric.tag].avg == 11

    def test_pool_keys_sorted(self):
        summary = asyncio.run(
            _summarize_specs(*(make_spec(s) for s in _TWO_RECORD_SPECS))
        )
        keys = list(summary.pooled_spec_decode_acceptance_histogram)
        assert keys == sorted(keys)

    def test_run_level_derived_values(self):
        summary = asyncio.run(
            _summarize_specs(*(make_spec(s) for s in _TWO_RECORD_SPECS))
        )
        results = summary.results
        assert results[TotalSpecDecodeStepsMetric.tag].avg == 11
        assert results[TotalAcceptedDraftTokensMetric.tag].avg == 20
        assert results[TotalDraftTokensMetric.tag].avg == 44
        # 1 + sum(accepted)/sum(steps) = 1 + 20/11
        assert results[
            SpecDecodeTokenWeightedAcceptanceLengthMetric.tag
        ].avg == pytest.approx(1 + 20 / 11)
        # sum(accepted)/sum(draft) * 100 = 20/44 * 100
        assert results[
            SpecDecodeOverallDraftAcceptanceRateMetric.tag
        ].avg == pytest.approx(20 / 44 * 100)

    def test_no_spec_decode_metrics_without_records(self):
        summary = asyncio.run(_summarize_specs(None))
        assert TotalSpecDecodeStepsMetric.tag not in summary.results
        assert SpecDecodeTokenWeightedAcceptanceLengthMetric.tag not in summary.results

    def test_warmup_records_excluded_from_profiling_pool(self):
        async def run():
            acc = MetricsAccumulator(make_benchmark_run())
            await acc.process_record(
                _spec_metric_records_data(0, make_spec([3, 3]))  # profiling
            )
            await acc.process_record(
                _spec_metric_records_data(
                    1, make_spec([1, 1, 1]), phase=CreditPhase.WARMUP
                )
            )
            return await acc.export_results(ExportContext(phase=CreditPhase.PROFILING))

        summary = asyncio.run(run())
        # Only the profiling record's histogram ({3:2}) is pooled.
        assert summary.pooled_spec_decode_acceptance_histogram == {3: 2}

    def test_pool_scopes_by_phase_index_and_reconciles(self):
        # Two profiling-phase INSTANCES (phase_index 0 and 1). A phase_index-
        # scoped export must pool only that instance and reconcile with its
        # masked total_spec_decode_steps; a phase-only export merges both.
        async def run():
            acc = MetricsAccumulator(make_benchmark_run())
            await acc.process_record(
                _spec_metric_records_data(0, make_spec([3, 3]), phase_index=0)
            )
            await acc.process_record(
                _spec_metric_records_data(1, make_spec([1, 1, 1]), phase_index=1)
            )
            merged = await acc.export_results(
                ExportContext(phase=CreditPhase.PROFILING)
            )
            inst0 = await acc.export_results(
                ExportContext(phase=CreditPhase.PROFILING, phase_index=0)
            )
            inst1 = await acc.export_results(
                ExportContext(phase=CreditPhase.PROFILING, phase_index=1)
            )
            return merged, inst0, inst1

        merged, inst0, inst1 = asyncio.run(run())
        # Phase-only export merges both instances.
        assert merged.pooled_spec_decode_acceptance_histogram == {1: 3, 3: 2}
        # Each phase_index-scoped export pools only its instance and reconciles
        # with that instance's masked total_spec_decode_steps.
        assert inst0.pooled_spec_decode_acceptance_histogram == {3: 2}
        assert (
            sum(inst0.pooled_spec_decode_acceptance_histogram.values())
            == inst0.results[TotalSpecDecodeStepsMetric.tag].avg
        )
        assert inst1.pooled_spec_decode_acceptance_histogram == {1: 3}
        assert (
            sum(inst1.pooled_spec_decode_acceptance_histogram.values())
            == inst1.results[TotalSpecDecodeStepsMetric.tag].avg
        )

    def test_no_histogram_without_spec_decode(self):
        async def run():
            acc = MetricsAccumulator(make_benchmark_run())
            await acc.process_record(_spec_metric_records_data(0, None))
            return await acc.summarize()

        summary = asyncio.run(run())
        assert summary.pooled_spec_decode_acceptance_histogram is None


class TestRecordsManagerHistogramSelection:
    """``_pooled_spec_decode_histogram`` picks the one accumulator that pooled a
    histogram. Only ``metric_results`` should, so more than one populated is a
    broken single-source invariant (a developer error from adding a second
    pooling accumulator) -- it warns and takes the first rather than silently
    picking a dict-ordering winner.
    """

    @staticmethod
    def _ctx(*histograms: dict[int, int] | None):
        return SimpleNamespace(
            accumulator_outputs={
                i: AccumulatorMetricsSummary(
                    results={}, pooled_spec_decode_acceptance_histogram=h
                )
                for i, h in enumerate(histograms)
            }
        )

    def test_single_populated_returned_without_warning(self):
        with patch("aiperf.records.records_manager._logger") as logger:
            result = _pooled_spec_decode_histogram(self._ctx(None, {0: 5}))
        assert result == {0: 5}
        logger.warning.assert_not_called()

    def test_none_when_no_populated_histogram(self):
        assert _pooled_spec_decode_histogram(self._ctx(None, None)) is None

    def test_multiple_populated_warns_and_takes_first(self):
        with patch("aiperf.records.records_manager._logger") as logger:
            result = _pooled_spec_decode_histogram(self._ctx({0: 5}, {1: 7}))
        assert result == {0: 5}
        logger.warning.assert_called_once()


class TestSummarySerialization:
    def test_to_json_includes_histogram_when_present(self):
        summary = AccumulatorMetricsSummary(
            results={}, pooled_spec_decode_acceptance_histogram={0: 3, 1: 1}
        )
        assert summary.to_json()["pooled_spec_decode_acceptance_histogram"] == {
            0: 3,
            1: 1,
        }

    def test_to_json_omits_histogram_when_absent(self):
        summary = AccumulatorMetricsSummary(results={})
        assert "pooled_spec_decode_acceptance_histogram" not in summary.to_json()


class TestConsoleHistogramRendering:
    def test_ticket_example_line(self):
        line = format_acceptance_histogram_line({0: 6100, 1: 300, 2: 1800, 3: 1800})
        assert line == (
            "Accepted drafts per step (% of steps):  0: 61%   1: 3%   2: 18%   3: 18%"
        )

    def test_overflow_folds_into_capped_bucket(self):
        histogram = {j: 1 for j in range(12)}  # buckets 0..11, 1 step each
        line = format_acceptance_histogram_line(histogram)
        assert f">={SPEC_DECODE_HISTOGRAM_CONSOLE_CAP}:" in line
        # Buckets 8..11 (4 of 12 steps) fold into the trailing bucket.
        assert ">=8: 33%" in line
        # Only buckets 0..7 plus the fold are shown -> 9 space-separated entries.
        body = line.split("):  ", 1)[1]
        assert len(body.split("   ")) == SPEC_DECODE_HISTOGRAM_CONSOLE_CAP + 1

    @pytest.mark.parametrize(
        "histogram",
        [param({}, id="empty"), param({0: 0, 1: 0}, id="all_zero")],
    )  # fmt: skip
    def test_empty_histogram_returns_none(self, histogram):
        assert format_acceptance_histogram_line(histogram) is None
