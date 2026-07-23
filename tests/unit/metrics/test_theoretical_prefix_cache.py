# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import (
    Conversation,
    ConversationMetadata,
    DatasetMetadata,
    ErrorDetails,
    MetricRecordMetadata,
    Turn,
    TurnMetadata,
)
from aiperf.metrics.theoretical_prefix_cache import (
    THEORETICAL_PREFIX_CACHE_HIT_TAG,
    TheoreticalPrefixCacheAccumulator,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    AccumulatorType,
    DatasetSamplingStrategy,
    EndpointType,
    PluginType,
)
from aiperf.records.records_manager_processing import load_accumulators
from tests.unit.conftest import make_benchmark_run


def _record(
    *,
    conversation_id: str,
    turn_index: int,
    error: ErrorDetails | None = None,
    benchmark_phase: CreditPhase = CreditPhase.PROFILING,
) -> MetricRecordsData:
    return MetricRecordsData(
        metadata=MetricRecordMetadata(
            session_num=turn_index,
            request_start_ns=1000 + turn_index,
            request_end_ns=2000 + turn_index,
            conversation_id=conversation_id,
            turn_index=turn_index,
            record_processor_id="rp",
            benchmark_phase=benchmark_phase,
            worker_id="worker",
        ),
        metrics={},
        error=error,
    )


def test_turn_metadata_carries_theoretical_prefix_counts() -> None:
    turn = Turn(
        theoretical_prefix_cache_hit_blocks=3,
        theoretical_prefix_cache_total_blocks=5,
    )

    metadata = turn.metadata()

    assert metadata.theoretical_prefix_cache_hit_blocks == 3
    assert metadata.theoretical_prefix_cache_total_blocks == 5


def test_conversation_metadata_carries_theoretical_prefix_counts() -> None:
    conversation = Conversation(
        session_id="trace-a",
        turns=[
            Turn(
                theoretical_prefix_cache_hit_blocks=3,
                theoretical_prefix_cache_total_blocks=5,
            )
        ],
    )

    [metadata] = conversation.metadata().turns

    assert metadata.theoretical_prefix_cache_hit_blocks == 3
    assert metadata.theoretical_prefix_cache_total_blocks == 5


def test_accumulator_reports_cumulative_theoretical_prefix_hit_rate() -> None:
    asyncio.run(_run_accumulator_reports_cumulative_theoretical_prefix_hit_rate())


async def _run_accumulator_reports_cumulative_theoretical_prefix_hit_rate() -> None:
    acc = TheoreticalPrefixCacheAccumulator(
        make_benchmark_run(endpoint_type=EndpointType.CHAT)
    )
    acc.on_dataset_configured(
        DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="trace-a",
                    turns=[
                        TurnMetadata(
                            theoretical_prefix_cache_hit_blocks=0,
                            theoretical_prefix_cache_total_blocks=3,
                        ),
                        TurnMetadata(
                            theoretical_prefix_cache_hit_blocks=3,
                            theoretical_prefix_cache_total_blocks=4,
                        ),
                    ],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
    )

    await acc.process_record(_record(conversation_id="trace-a", turn_index=0))
    await acc.process_record(_record(conversation_id="trace-a", turn_index=1))

    [result] = await acc.summarize()
    assert result.tag == THEORETICAL_PREFIX_CACHE_HIT_TAG
    assert result.current == pytest.approx(100.0 * 3 / 7)
    assert result.avg == pytest.approx(100.0 * 3 / 7)
    assert result.count == 7
    assert result.sum == 3


def test_accumulator_skips_missing_metadata_and_errors() -> None:
    asyncio.run(_run_accumulator_skips_missing_metadata_and_errors())


async def _run_accumulator_skips_missing_metadata_and_errors() -> None:
    acc = TheoreticalPrefixCacheAccumulator(
        make_benchmark_run(endpoint_type=EndpointType.CHAT)
    )
    acc.on_dataset_configured(
        DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="trace-a",
                    turns=[
                        TurnMetadata(
                            theoretical_prefix_cache_hit_blocks=1,
                            theoretical_prefix_cache_total_blocks=2,
                        )
                    ],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
    )

    await acc.process_record(_record(conversation_id="missing", turn_index=0))
    await acc.process_record(
        _record(
            conversation_id="trace-a",
            turn_index=0,
            error=ErrorDetails(message="bad request"),
        )
    )

    assert await acc.summarize() == []


def test_export_results_scopes_to_profiling_phase() -> None:
    """Warmup must not bleed into the profiling headline hit rate.

    Equal block totals (10 each) make all-phases average 50% while profiling
    alone is 10% — the bug this guards against.
    """
    asyncio.run(_run_export_results_scopes_to_profiling_phase())


async def _run_export_results_scopes_to_profiling_phase() -> None:
    acc = TheoreticalPrefixCacheAccumulator(
        make_benchmark_run(endpoint_type=EndpointType.CHAT)
    )
    acc.on_dataset_configured(
        DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="trace-a",
                    turns=[
                        # Warmup: 9/10 = 90%
                        TurnMetadata(
                            theoretical_prefix_cache_hit_blocks=9,
                            theoretical_prefix_cache_total_blocks=10,
                        ),
                        # Profiling: 1/10 = 10%
                        TurnMetadata(
                            theoretical_prefix_cache_hit_blocks=1,
                            theoretical_prefix_cache_total_blocks=10,
                        ),
                    ],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
    )

    await acc.process_record(
        _record(
            conversation_id="trace-a",
            turn_index=0,
            benchmark_phase=CreditPhase.WARMUP,
        )
    )
    await acc.process_record(
        _record(
            conversation_id="trace-a",
            turn_index=1,
            benchmark_phase=CreditPhase.PROFILING,
        )
    )

    assert acc.supports_phase_scoped_export is True

    [profiling] = await acc.export_results(ExportContext(phase=CreditPhase.PROFILING))
    assert profiling.current == pytest.approx(10.0)
    assert profiling.avg == pytest.approx(10.0)
    assert profiling.count == 10
    assert profiling.sum == 1

    [warmup] = await acc.export_results(ExportContext(phase=CreditPhase.WARMUP))
    assert warmup.current == pytest.approx(90.0)

    # summarize() remains phase-agnostic for callers that still use it.
    [all_phases] = await acc.summarize()
    assert all_phases.current == pytest.approx(50.0)


def test_theoretical_prefix_cache_registered_as_accumulator_plugin() -> None:
    """Port regression: accumulator must be in plugins.yaml so RecordsManager loads it."""
    names = [e.name for e in plugins.iter_entries(PluginType.ACCUMULATOR)]
    assert "theoretical_prefix_cache" in names
    assert AccumulatorType.THEORETICAL_PREFIX_CACHE == "theoretical_prefix_cache"
    cls = plugins.get_class(
        PluginType.ACCUMULATOR, AccumulatorType.THEORETICAL_PREFIX_CACHE
    )
    assert cls is TheoreticalPrefixCacheAccumulator
    entry = plugins.get_entry(PluginType.ACCUMULATOR, "theoretical_prefix_cache")
    assert entry.metadata is not None
    assert entry.metadata.get("record_types") == ["metric_records"]


def test_load_accumulators_includes_theoretical_prefix_cache(
    benchmark_run,
) -> None:
    """RecordsManager loader must construct the registered theoretical prefix accumulator."""
    from unittest.mock import MagicMock

    host = MagicMock()
    host.service_id = "records-manager"
    host.run = benchmark_run
    host.pub_client = MagicMock()
    host.attach_child_lifecycle = MagicMock()
    host.debug = MagicMock()
    host.error = MagicMock()

    accumulators = load_accumulators(host)

    assert AccumulatorType.THEORETICAL_PREFIX_CACHE in accumulators
    acc = accumulators[AccumulatorType.THEORETICAL_PREFIX_CACHE]
    assert isinstance(acc, TheoreticalPrefixCacheAccumulator)
    host.error.assert_not_called()
