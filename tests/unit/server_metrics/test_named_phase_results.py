# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete named-phase identity remains intact in manager-owned aggregation."""

import pytest

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import CreditPhase, PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.accumulator import ServerMetricsAccumulator
from aiperf.timing.config import AGENTIC_WARMUP_PHASE_NAME
from tests.unit.conftest import make_run_from_cli


def _record(
    timestamp_ns: int,
    *,
    phase_index: int,
    profiling_index: int,
    phase_name: str,
) -> ServerMetricsRecord:
    return ServerMetricsRecord(
        endpoint_url="http://server:8000/metrics",
        timestamp_ns=timestamp_ns,
        benchmark_phase=CreditPhase.PROFILING,
        phase_index=phase_index,
        profiling_index=profiling_index,
        phase_name=phase_name,
        phase_kind="profiling",
        metrics={
            "vllm:num_requests_running": MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="running",
                samples=[MetricSample(value=float(timestamp_ns))],
            )
        },
    )


def _warmup_record(
    timestamp_ns: int, value: float, *, phase_instance_id: int | None = None
) -> ServerMetricsRecord:
    """A synthesized AGENTIC_REPLAY warmup scrape: phase_index is always None."""
    return ServerMetricsRecord(
        endpoint_url="http://server:8000/metrics",
        timestamp_ns=timestamp_ns,
        benchmark_phase=CreditPhase.WARMUP,
        phase_index=None,
        profiling_index=None,
        phase_name=AGENTIC_WARMUP_PHASE_NAME,
        phase_kind="warmup",
        phase_instance_id=phase_instance_id,
        metrics={
            "vllm:num_requests_running": MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="running",
                samples=[MetricSample(value=value)],
            )
        },
    )


@pytest.mark.asyncio
async def test_export_results_contains_exact_named_phase_summaries() -> None:
    accumulator = ServerMetricsAccumulator(
        run=make_run_from_cli(
            CLIConfig(
                model_names=["model"],
                endpoint_type=EndpointType.CHAT,
                urls=["http://server:8000/v1/chat/completions"],
            )
        )
    )
    for record in (
        _record(10, phase_index=0, profiling_index=0, phase_name="baseline"),
        _record(20, phase_index=0, profiling_index=0, phase_name="baseline"),
        _record(30, phase_index=1, profiling_index=1, phase_name="main"),
        _record(40, phase_index=1, profiling_index=1, phase_name="main"),
    ):
        await accumulator.process_record(record)

    results = await accumulator.export_results(ExportContext(start_ns=10, end_ns=41))

    assert results is not None
    assert [result.phase_name for result in results.phase_results] == [
        "baseline",
        "main",
    ]
    assert [result.phase_index for result in results.phase_results] == [0, 1]
    assert all(result.endpoint_summaries for result in results.phase_results)


@pytest.mark.asyncio
async def test_export_results_keeps_repeated_agentic_warmup_instances_distinct() -> (
    None
):
    """Two synthesized agentic-warmup instances (phase_index=None) must not pool.

    AGENTIC_REPLAY synthesizes a warmup phase with phase_index=None for every
    instance (it isn't a declared cfg.phases entry -- see
    _build_agentic_warmup_config). A multi-phase agentic plan can run this
    synthesized warmup more than once with the identical (None, phase_name)
    identity; each instance must still produce its own phase_results entry
    instead of collapsing into one pooled result.
    """
    accumulator = ServerMetricsAccumulator(
        run=make_run_from_cli(
            CLIConfig(
                model_names=["model"],
                endpoint_type=EndpointType.CHAT,
                urls=["http://server:8000/v1/chat/completions"],
            )
        )
    )
    for record in (
        _warmup_record(10, 1.0),
        _warmup_record(20, 2.0),
        # An intervening profiling phase marks the boundary between the two
        # warmup instances (mirrors a real multi-phase agentic run).
        _record(30, phase_index=0, profiling_index=0, phase_name="main"),
        _warmup_record(40, 3.0),
        _warmup_record(50, 4.0),
    ):
        await accumulator.process_record(record)

    results = await accumulator.export_results(ExportContext(start_ns=10, end_ns=51))

    assert results is not None
    warmup_results = [
        result
        for result in results.phase_results
        if result.phase_name == AGENTIC_WARMUP_PHASE_NAME
    ]
    assert len(warmup_results) == 2, (
        "expected two distinct agentic-warmup phase_results entries, "
        f"got {len(warmup_results)}"
    )
    assert all(result.phase_index is None for result in warmup_results)
    assert [result.start_ns for result in warmup_results] == [10, 40]
    assert [result.end_ns for result in warmup_results] == [20, 50]


@pytest.mark.asyncio
async def test_synthetic_storage_indexes_do_not_mutate_public_phase_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic warmup indexes are storage metadata, not model copies."""

    def fail_model_copy(*_args, **_kwargs):
        raise AssertionError(
            "storage must not bypass phase_index validation via model_copy"
        )

    monkeypatch.setattr(ServerMetricsRecord, "model_copy", fail_model_copy)
    accumulator = ServerMetricsAccumulator(
        run=make_run_from_cli(
            CLIConfig(
                model_names=["model"],
                endpoint_type=EndpointType.CHAT,
                urls=["http://server:8000/v1/chat/completions"],
            )
        )
    )
    first = _warmup_record(10, 1.0, phase_instance_id=1)
    second = _warmup_record(20, 2.0, phase_instance_id=2)

    await accumulator.process_server_metrics_record(first)
    await accumulator.process_server_metrics_record(second)

    assert first.phase_index is None
    assert second.phase_index is None
    time_series = accumulator.get_hierarchy_for_export().endpoints[first.endpoint_url]
    data = next(iter(time_series.metrics.values())).data
    stored_indexes = data._phase_indices[: data._size]
    assert len(set(stored_indexes)) == 2
    assert all(index < 0 for index in stored_indexes)


@pytest.mark.asyncio
async def test_interleaved_scrapes_keep_one_warmup_instance_together() -> None:
    """Overlapping scrapes must not split a single warmup instance.

    Scrapes are dispatched fire-and-forget (`_collect_metrics_loop` uses
    execute_async precisely so a slow scrape doesn't delay the next), and each
    one carries the phase snapshot taken at *its own* start. When a scrape
    outlives a phase boundary its records arrive interleaved with the next
    phase's, so arrival order is not a phase-identity signal.

    Keying synthesized indices off "the signature changed from the previous
    record" therefore minted a fresh index every time the interleaving
    alternated, shattering one warmup instance into a phase_results entry per
    record. `phase_instance_id`, stamped once per CREDIT_PHASE_START, is
    order-independent and holds the instance together.
    """
    accumulator = ServerMetricsAccumulator(
        run=make_run_from_cli(
            CLIConfig(
                model_names=["model"],
                endpoint_type=EndpointType.CHAT,
                urls=["http://server:8000/v1/chat/completions"],
            )
        )
    )
    # One warmup instance (id=1) whose scrapes straddle the start of profiling,
    # so its records arrive interleaved with the profiling phase's.
    for record in (
        _warmup_record(10, 1.0, phase_instance_id=1),
        _record(20, phase_index=0, profiling_index=0, phase_name="main"),
        _warmup_record(30, 2.0, phase_instance_id=1),
        _record(40, phase_index=0, profiling_index=0, phase_name="main"),
        _warmup_record(50, 3.0, phase_instance_id=1),
    ):
        await accumulator.process_record(record)

    results = await accumulator.export_results(ExportContext(start_ns=10, end_ns=51))

    assert results is not None
    warmup_results = [
        result
        for result in results.phase_results
        if result.phase_name == AGENTIC_WARMUP_PHASE_NAME
    ]
    assert len(warmup_results) == 1, (
        "one warmup instance must stay one phase_results entry regardless of "
        f"how its scrapes interleave, got {len(warmup_results)}"
    )


@pytest.mark.asyncio
async def test_stamped_instance_ids_separate_repeated_warmups() -> None:
    """Distinct occurrences stay distinct even when their records interleave."""
    accumulator = ServerMetricsAccumulator(
        run=make_run_from_cli(
            CLIConfig(
                model_names=["model"],
                endpoint_type=EndpointType.CHAT,
                urls=["http://server:8000/v1/chat/completions"],
            )
        )
    )
    for record in (
        _warmup_record(10, 1.0, phase_instance_id=1),
        _warmup_record(20, 2.0, phase_instance_id=3),
        _warmup_record(30, 3.0, phase_instance_id=1),
        _warmup_record(40, 4.0, phase_instance_id=3),
    ):
        await accumulator.process_record(record)

    results = await accumulator.export_results(ExportContext(start_ns=10, end_ns=41))

    assert results is not None
    warmup_results = [
        result
        for result in results.phase_results
        if result.phase_name == AGENTIC_WARMUP_PHASE_NAME
    ]
    assert len(warmup_results) == 2


def _labelled_warmup_record(
    timestamp_ns: int, *, phase_name: str, phase_instance_id: int
) -> ServerMetricsRecord:
    """A declared (indexed) warmup scrape carrying an explicit display label."""
    return ServerMetricsRecord(
        endpoint_url="http://server:8000/metrics",
        timestamp_ns=timestamp_ns,
        benchmark_phase=CreditPhase.WARMUP,
        phase_index=0,
        profiling_index=None,
        phase_name=phase_name,
        phase_kind="warmup",
        phase_instance_id=phase_instance_id,
        metrics={
            "vllm:num_requests_running": MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="running",
                samples=[MetricSample(value=float(timestamp_ns))],
            )
        },
    )


@pytest.mark.asyncio
async def test_divergent_phase_labels_split_one_occurrence() -> None:
    """Export captures are keyed by display label, so labels must be unified upstream.

    ``_phase_captures`` is keyed on ``(sample_phase_index, phase_name)``. Two
    scrapes of the same phase occurrence that disagree on the cosmetic label
    therefore become two exported ``ServerMetricsResults`` -- and because both
    captures filter samples on the same ``sample_phase_index``, the two results
    summarize the *same* samples under different names.

    That is why ``ServerMetricsManager._stamped_like_active`` carries the active
    phase's ``phase_name`` onto rebuilt scrape identities and not just the
    occurrence id: the two publishers of a phase with no configured name fall
    back to different labels (``"warmup"`` vs ``"warmup_0"``). This test pins the
    accumulator behavior that makes that unification load-bearing.
    """
    accumulator = ServerMetricsAccumulator(
        run=make_run_from_cli(
            CLIConfig(
                model_names=["model"],
                endpoint_type=EndpointType.CHAT,
                urls=["http://server:8000/v1/chat/completions"],
            )
        )
    )
    for record in (
        _labelled_warmup_record(10, phase_name="warmup", phase_instance_id=1),
        _labelled_warmup_record(20, phase_name="warmup", phase_instance_id=1),
        _labelled_warmup_record(30, phase_name="warmup_0", phase_instance_id=1),
    ):
        await accumulator.process_record(record)

    results = await accumulator.export_results(ExportContext(start_ns=10, end_ns=31))

    assert results is not None
    assert [result.phase_name for result in results.phase_results] == [
        "warmup",
        "warmup_0",
    ]

    unified = ServerMetricsAccumulator(
        run=make_run_from_cli(
            CLIConfig(
                model_names=["model"],
                endpoint_type=EndpointType.CHAT,
                urls=["http://server:8000/v1/chat/completions"],
            )
        )
    )
    for record in (
        _labelled_warmup_record(10, phase_name="warmup", phase_instance_id=1),
        _labelled_warmup_record(20, phase_name="warmup", phase_instance_id=1),
        _labelled_warmup_record(30, phase_name="warmup", phase_instance_id=1),
    ):
        await unified.process_record(record)

    unified_results = await unified.export_results(
        ExportContext(start_ns=10, end_ns=31)
    )
    assert unified_results is not None
    assert [result.phase_name for result in unified_results.phase_results] == ["warmup"]
