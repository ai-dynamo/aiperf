# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speculative-decoding acceptance metrics.

Every metric here reads only the engine-neutral
``ParsedResponseRecord.spec_decode_acceptance`` record (a
``SpecDecodeAcceptanceRecord`` produced by an engine adapter) and never branches
on the serving engine. When the record is absent -- spec decode is off, or the
request had no verify steps -- each metric raises ``NoMetricValue`` so it drops
out of the run cleanly and nothing spec-decode-related is shown or exported.

The pooled accepted-draft histogram is not a metric: dict aggregation lives
outside the scalar/list metric machinery, in ``MetricsAccumulator`` (pooled per
phase) and surfaces on ``ProfileResults.pooled_spec_decode_acceptance_histogram``.
"""

from typing import ClassVar, Generic

from aiperf.common.enums import (
    GenericMetricUnit,
    MetricConsoleGroup,
    MetricFlags,
    MetricValueTypeVarT,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.models.spec_decode_models import SpecDecodeAcceptanceRecord
from aiperf.metrics import BaseDerivedMetric, BaseRecordMetric
from aiperf.metrics.derived_sum_metric import DerivedSumMetric
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict

_MISSING = (
    "No speculative-decoding acceptance record on this request (spec decode off, "
    "the request had no verify steps, or the engine reported no stats)."
)


class BaseSpecDecodeRecordMetric(
    BaseRecordMetric[MetricValueTypeVarT], Generic[MetricValueTypeVarT]
):
    """Reads one field from ``record.spec_decode_acceptance``.

    Mirrors ``BaseUsageRecordMetric``: subclasses set ``spec_decode_field`` (the
    attribute name on ``SpecDecodeAcceptanceRecord``) and inherit the None-check
    that raises ``NoMetricValue`` when the record is absent. Subclasses that
    compute a value from several fields override ``_parse_record`` and call
    ``_record`` for the same None-check.
    """

    # The base class itself is not registerable -- it carries no tag. Subclasses
    # flip this in __init_subclass__ so they register normally.
    __is_abstract__: ClassVar[bool] = True

    spec_decode_field: ClassVar[str | None] = None
    """Attribute on ``SpecDecodeAcceptanceRecord`` read by the default
    ``_parse_record``. None on subclasses that override ``_parse_record``."""

    def __init_subclass__(cls, **kwargs) -> None:
        cls.__is_abstract__ = False
        return super().__init_subclass__(**kwargs)

    def _record(self, record: ParsedResponseRecord) -> SpecDecodeAcceptanceRecord:
        """Return the neutral record or raise ``NoMetricValue`` when absent."""
        spec = record.spec_decode_acceptance
        if spec is None:
            raise NoMetricValue(_MISSING)
        return spec

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> MetricValueTypeVarT:
        if not self.spec_decode_field:
            raise TypeError(
                f"{type(self).__name__} must set spec_decode_field or override "
                "_parse_record"
            )
        return getattr(self._record(record), self.spec_decode_field)


class SpecDecodeAcceptanceLengthMetric(BaseSpecDecodeRecordMetric[float]):
    """Per-request mean acceptance length: tokens emitted per verify step
    including the always-accepted bonus token (``1 + accepted / steps``, the
    ``j + 1`` acceptance length).

    Reported alongside the token-weighted variant
    (``spec_decode_token_weighted_acceptance_length``): this one averages
    per-request means equally while the token-weighted one weights by verify
    steps, so the two can diverge when per-request step counts vary and correlate
    with acceptance, and coincide when every request runs a similar number of
    steps.
    """

    tag = "spec_decode_acceptance_length"
    header = "Acceptance Length"
    short_header = "Acceptance Length"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    flags = MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.SPEC_DECODE
    display_order = 5000
    required_metrics = None

    spec_decode_field = "mean_acceptance_length"


class SpecDecodeDraftAcceptanceRateMetric(BaseSpecDecodeRecordMetric[float]):
    """Per-request draft acceptance rate: accepted draft tokens over proposed
    draft tokens (``num_accepted_draft_tokens / num_draft_tokens``), as a
    percentage. Draft-only -- excludes the bonus token.
    """

    tag = "spec_decode_draft_acceptance_rate"
    header = "Draft Acceptance Rate"
    short_header = "Draft Accept Rate"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PERCENT
    flags = MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.SPEC_DECODE
    display_order = 5020
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        # The record stores draft_acceptance_rate as a 0..1 fraction; scale to a
        # percent so the console/JSON column reads 0..100 like other rate metrics.
        return self._record(record).draft_acceptance_rate * 100.0


class SpecDecodeAcceptedPerVerifiedMetric(BaseSpecDecodeRecordMetric[float]):
    """Per-request accepted-per-verified ratio: emitted tokens (accepted drafts
    plus one bonus per step) over verified tokens (proposed drafts plus one
    bonus per step) = ``(num_accepted + num_steps) / (num_draft + num_steps)``
    = ``(j + 1) / (l + 1)``.
    """

    tag = "spec_decode_accepted_per_verified"
    header = "Accepted per Verified"
    short_header = "Accepted / Verified"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    flags = MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.SPEC_DECODE
    display_order = 5030
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        spec = self._record(record)
        denominator = spec.num_draft_tokens + spec.num_spec_steps
        if denominator == 0:
            raise NoMetricValue(
                "Accepted-per-verified is undefined with no drafts and no verify steps."
            )
        return (spec.num_accepted_draft_tokens + spec.num_spec_steps) / denominator


class SpecDecodeStepsMetric(BaseSpecDecodeRecordMetric[int]):
    """Per-request number of speculative verification steps
    (``num_spec_steps``). Equals the sum of the request's acceptance histogram
    counts.
    """

    tag = "spec_decode_steps"
    header = "Spec Decode Steps"
    short_header = "Spec Decode Steps"
    short_header_hide_unit = True
    unit = GenericMetricUnit.COUNT
    console_group = MetricConsoleGroup.SPEC_DECODE
    display_order = 5040
    required_metrics = None

    spec_decode_field = "num_spec_steps"


class SpecDecodeAcceptedDraftTokensMetric(BaseSpecDecodeRecordMetric[int]):
    """Per-request accepted draft tokens (``num_accepted_draft_tokens``),
    excluding bonus tokens. Exported, not shown in the console.
    """

    tag = "spec_decode_accepted_draft_tokens"
    header = "Accepted Draft Tokens"
    short_header = "Accepted Draft"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    console_group = MetricConsoleGroup.NONE
    display_order = 5050
    required_metrics = None

    spec_decode_field = "num_accepted_draft_tokens"


class SpecDecodeDraftTokensMetric(BaseSpecDecodeRecordMetric[int]):
    """Per-request proposed draft tokens counted toward acceptance
    (``num_draft_tokens``). Exported, not shown in the console.
    """

    tag = "spec_decode_draft_tokens"
    header = "Draft Tokens"
    short_header = "Draft Tokens"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    console_group = MetricConsoleGroup.NONE
    display_order = 5060
    required_metrics = None

    spec_decode_field = "num_draft_tokens"


class TotalSpecDecodeStepsMetric(DerivedSumMetric[int, SpecDecodeStepsMetric]):
    """Total speculative verification steps across all requests
    (``Sum(num_spec_steps)``). Equals the sum of the pooled acceptance
    histogram counts.
    """

    tag = "total_spec_decode_steps"
    header = "Total Spec Decode Steps"
    short_header = "Total Spec Decode Steps"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.NONE
    display_order = 5140


class TotalAcceptedDraftTokensMetric(
    DerivedSumMetric[int, SpecDecodeAcceptedDraftTokensMetric]
):
    """Total accepted draft tokens across all requests
    (``Sum(num_accepted_draft_tokens)``)."""

    tag = "total_accepted_draft_tokens"
    header = "Total Accepted Draft Tokens"
    short_header = "Total Accepted Draft"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.NONE
    display_order = 5150


class TotalDraftTokensMetric(DerivedSumMetric[int, SpecDecodeDraftTokensMetric]):
    """Total proposed draft tokens across all requests
    (``Sum(num_draft_tokens)``)."""

    tag = "total_draft_tokens"
    header = "Total Draft Tokens"
    short_header = "Total Draft"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.NONE
    display_order = 5160


class SpecDecodeTokenWeightedAcceptanceLengthMetric(BaseDerivedMetric[float]):
    """Run-level token-weighted mean acceptance length: ``1 + Sum(accepted) /
    Sum(steps)``.

    Reported alongside the per-request-mean ``spec_decode_acceptance_length``:
    this weights every verify step equally while the per-request mean weights
    every request equally, so the two can diverge when requests vary in verify-step
    count and that count correlates with per-request acceptance.
    """

    tag = "spec_decode_token_weighted_acceptance_length"
    header = "Token-Weighted Acceptance Length"
    short_header = "Token-Wtd Accept Len"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    flags = MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.SPEC_DECODE
    display_order = 5010
    required_metrics = {
        TotalAcceptedDraftTokensMetric.tag,
        TotalSpecDecodeStepsMetric.tag,
    }

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        steps = metric_results.get_or_raise(TotalSpecDecodeStepsMetric)
        if steps == 0:
            raise NoMetricValue(
                "Token-weighted acceptance length is undefined with zero verify steps."
            )
        accepted = metric_results.get_or_raise(TotalAcceptedDraftTokensMetric)
        return 1.0 + accepted / steps


class SpecDecodeOverallDraftAcceptanceRateMetric(BaseDerivedMetric[float]):
    """Run-level (token-weighted) draft acceptance rate: ``Sum(accepted) /
    Sum(draft)``, as a percentage.

    Divides summed accepted draft tokens by summed proposed draft tokens across
    the whole run, so large and small requests contribute in proportion to
    their draft volume -- unlike the per-request
    ``spec_decode_draft_acceptance_rate`` average.
    """

    tag = "spec_decode_overall_draft_acceptance_rate"
    header = "Overall Draft Acceptance Rate"
    short_header = "Overall Draft Accept Rate"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PERCENT
    flags = MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.SPEC_DECODE
    display_order = 5025
    required_metrics = {
        TotalAcceptedDraftTokensMetric.tag,
        TotalDraftTokensMetric.tag,
    }

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        draft = metric_results.get_or_raise(TotalDraftTokensMetric)
        if draft == 0:
            raise NoMetricValue(
                "Overall draft acceptance rate is undefined with zero proposed "
                "draft tokens."
            )
        accepted = metric_results.get_or_raise(TotalAcceptedDraftTokensMetric)
        return (accepted / draft) * 100.0
