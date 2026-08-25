# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes progress handling for user-named benchmark phases."""

from __future__ import annotations

from unittest.mock import AsyncMock

import kopf
import pytest

from aiperf.api.routers.progress import _build_progress_annotations
from aiperf.common.enums import CreditPhase, SystemState
from aiperf.common.mixins.progress_tracker_mixin import (
    CombinedPhaseStats,
    ProgressTracker,
)
from aiperf.common.models import CreditPhaseStats
from aiperf.common.models.record_models import MetricRecordMetadata
from aiperf.common.types import PhaseKind
from aiperf.kubernetes.constants import ProgressAnnotations
from aiperf.operator.progress_client import ProgressClient
from aiperf.operator.progress_models import JobProgress
from aiperf.operator.status import StatusBuilder
from aiperf.records.records_manager import RecordsManager
from aiperf.records.records_tracker import RecordsTracker


def _stats(
    name: str,
    kind: PhaseKind,
    *,
    start_ns: int,
    complete: bool = False,
    phase_index: int = 0,
    profiling_index: int | None = 0,
) -> CombinedPhaseStats:
    """Build progress stats with separate phase identity and semantic role."""
    end_ns = start_ns + 10 if complete else None
    return CombinedPhaseStats(
        phase=CreditPhase(kind),
        phase_index=phase_index,
        profiling_index=profiling_index if kind == "profiling" else None,
        phase_name=name,
        phase_kind=kind,
        start_ns=start_ns,
        sent_end_ns=end_ns,
        requests_end_ns=end_ns,
        records_end_ns=end_ns,
        requests_sent=10,
        requests_completed=10,
        success_records=10,
        total_expected_requests=10,
        last_update_ns=start_ns + 5,
    )


def _status_builder() -> tuple[StatusBuilder, kopf.Patch]:
    patch = kopf.Patch()
    return StatusBuilder(patch, {}), patch


class TestNamedPhaseProducer:
    """The progress API producer must not collapse phases with the same kind."""

    def test_tracker_keys_multiple_profiling_phases_by_name(self) -> None:
        tracker = ProgressTracker()

        for phase_index, name in enumerate(("baseline", "steady_state_profile")):
            tracker.update_requests_stats(
                CreditPhaseStats(
                    phase=CreditPhase.PROFILING,
                    phase_index=phase_index,
                    profiling_index=phase_index,
                    phase_name=name,
                    phase_kind="profiling",
                    start_ns=phase_index + 1,
                )
            )

        assert set(tracker._phases) == {"baseline", "steady_state_profile"}
        assert tracker._phases["baseline"].phase_index == 0
        assert tracker._phases["steady_state_profile"].phase_index == 1

    def test_parser_preserves_custom_name_without_coercing_it_to_credit_phase(
        self,
    ) -> None:
        stats = _stats("main", "profiling", start_ns=10)

        progress = ProgressClient()._parse_progress_response(
            {
                "phases": {"main": stats.model_dump(mode="json")},
                "system_state": "profiling",
            }
        )

        assert set(progress.phases) == {"main"}
        assert progress.phases["main"].phase == CreditPhase.PROFILING
        assert progress.phases["main"].phase_kind == "profiling"

    def test_records_report_updates_same_named_entry_without_legacy_key(self) -> None:
        request_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            phase_index=2,
            profiling_index=1,
            phase_name="steady_state_profile",
            phase_kind="profiling",
            start_ns=10,
            requests_end_ns=20,
            final_requests_completed=1,
            total_expected_requests=1,
        )
        records_tracker = RecordsTracker()
        records_tracker.update_phase_info(request_stats)
        records_tracker.update_from_request(
            MetricRecordMetadata(
                session_num=0,
                request_start_ns=11,
                request_end_ns=19,
                worker_id="worker-1",
                record_processor_id="processor-1",
                benchmark_phase=CreditPhase.PROFILING,
                phase_index=2,
            ),
            None,
        )
        assert records_tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING, phase_index=2
        )

        progress_tracker = ProgressTracker()
        progress_tracker.update_requests_stats(request_stats)
        progress_tracker.update_records_stats(
            records_tracker.create_stats_for_phase(CreditPhase.PROFILING)
        )

        assert set(progress_tracker._phases) == {"steady_state_profile"}
        progress = JobProgress(
            phases=progress_tracker._phases,
            results_exported=True,
        )
        assert progress.primary_phase == "steady_state_profile"
        assert progress.primary_phase_stats is not None
        assert progress.primary_phase_stats.is_records_complete
        assert progress.is_complete

    def test_records_progress_stats_preserve_unindexed_legacy_phase(self) -> None:
        records_tracker = RecordsTracker()
        records_tracker.update_phase_info(
            CreditPhaseStats(
                phase=CreditPhase.PROFILING,
                start_ns=10,
                requests_end_ns=20,
                final_requests_completed=1,
                total_expected_requests=1,
            )
        )
        records_tracker.update_from_request(
            MetricRecordMetadata(
                session_num=0,
                request_start_ns=11,
                request_end_ns=19,
                worker_id="worker-1",
                record_processor_id="processor-1",
                benchmark_phase=CreditPhase.PROFILING,
            ),
            None,
        )
        assert records_tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
        )

        stats = records_tracker.create_progress_stats_for_phase(CreditPhase.PROFILING)

        assert len(stats) == 1
        assert stats[0].phase_index is None
        assert stats[0].phase_name is None
        assert stats[0].is_records_complete

    @pytest.mark.asyncio
    async def test_periodic_records_report_publishes_every_named_profiling_phase(
        self,
    ) -> None:
        records_tracker = RecordsTracker()
        for phase_index, name in enumerate(("baseline", "steady_state_profile")):
            request_stats = CreditPhaseStats(
                phase=CreditPhase.PROFILING,
                phase_index=phase_index,
                profiling_index=phase_index,
                phase_name=name,
                phase_kind="profiling",
                start_ns=phase_index + 1,
                requests_end_ns=phase_index + 10,
                final_requests_completed=1,
                total_expected_requests=1,
            )
            records_tracker.update_phase_info(request_stats)
            records_tracker.update_from_request(
                MetricRecordMetadata(
                    session_num=phase_index,
                    request_start_ns=phase_index + 2,
                    request_end_ns=phase_index + 9,
                    worker_id="worker-1",
                    record_processor_id="processor-1",
                    benchmark_phase=CreditPhase.PROFILING,
                    phase_index=phase_index,
                ),
                None,
            )
            assert records_tracker.check_and_set_all_records_received_for_phase(
                CreditPhase.PROFILING, phase_index=phase_index
            )

        manager = RecordsManager.__new__(RecordsManager)
        manager._records_tracker = records_tracker
        manager._publish_processing_stats = AsyncMock()

        await manager._report_records_task()

        progress_tracker = ProgressTracker()
        published_stats = [
            call.args[0] for call in manager._publish_processing_stats.await_args_list
        ]
        for stats in published_stats:
            progress_tracker.update_records_stats(stats)

        assert [stats.phase_name for stats in published_stats] == [
            "baseline",
            "steady_state_profile",
        ]
        assert set(progress_tracker._phases) == {"baseline", "steady_state_profile"}
        assert "profiling" not in progress_tracker._phases
        assert all(
            phase.is_records_complete for phase in progress_tracker._phases.values()
        )

    @pytest.mark.asyncio
    async def test_terminal_records_report_publishes_each_named_warmup_phase(
        self,
    ) -> None:
        records_tracker = RecordsTracker()
        manager = RecordsManager.__new__(RecordsManager)
        manager._records_tracker = records_tracker
        manager._publish_processing_stats = AsyncMock()
        manager._handle_all_records_received = AsyncMock()
        manager._all_records_received_phases = set()

        for phase_index, name in enumerate(("cache_prime", "cooldown")):
            records_tracker.update_phase_info(
                CreditPhaseStats(
                    phase=CreditPhase.WARMUP,
                    phase_index=phase_index,
                    phase_name=name,
                    phase_kind="warmup",
                    start_ns=phase_index + 1,
                    requests_end_ns=phase_index + 10,
                    final_requests_completed=1,
                    total_expected_requests=1,
                )
            )
            records_tracker.update_from_request(
                MetricRecordMetadata(
                    session_num=phase_index,
                    request_start_ns=phase_index + 2,
                    request_end_ns=phase_index + 9,
                    worker_id="worker-1",
                    record_processor_id="processor-1",
                    benchmark_phase=CreditPhase.WARMUP,
                    phase_index=phase_index,
                ),
                None,
            )
            assert records_tracker.check_and_set_all_records_received_for_phase(
                CreditPhase.WARMUP
            )
            await manager._handle_all_records_received_once(CreditPhase.WARMUP)

        latest_stats = {
            call.args[0].phase_name: call.args[0]
            for call in manager._publish_processing_stats.await_args_list
        }
        assert set(latest_stats) == {"cache_prime", "cooldown"}
        assert all(stats.is_records_complete for stats in latest_stats.values())
        manager._handle_all_records_received.assert_awaited_once_with(
            CreditPhase.WARMUP
        )


class TestNamedPhaseSemantics:
    """Phase kind, not phase name, controls K8s lifecycle behavior."""

    def test_custom_profiling_name_is_primary_and_completes(self) -> None:
        progress = JobProgress(
            results_exported=True,
            phases={
                "cache_prime": _stats(
                    "cache_prime",
                    "warmup",
                    start_ns=1,
                    complete=True,
                    profiling_index=None,
                ),
                "steady_state_profile": _stats(
                    "steady_state_profile", "profiling", start_ns=20, complete=True
                ),
            },
        )

        assert progress.primary_phase == "steady_state_profile"
        assert progress.current_phase == "steady_state_profile"
        assert progress.is_benchmark_phase_active is True
        assert progress.is_complete is True

    def test_custom_warmup_name_is_excluded_from_completion(self) -> None:
        progress = JobProgress(
            results_exported=True,
            phases={
                "cache_prime": _stats(
                    "cache_prime",
                    "warmup",
                    start_ns=10,
                    complete=True,
                    profiling_index=None,
                )
            },
        )

        assert progress.current_phase == "cache_prime"
        assert progress.primary_phase is None
        assert progress.is_benchmark_phase_active is False
        assert progress.is_complete is False

    def test_legacy_aggregate_does_not_override_concrete_named_phase(self) -> None:
        progress = JobProgress(
            results_exported=True,
            phases={
                "main": _stats("main", "profiling", start_ns=10),
                "profiling": CombinedPhaseStats(
                    phase=CreditPhase.PROFILING,
                    start_ns=100,
                    requests_end_ns=110,
                    records_end_ns=120,
                ),
            },
        )

        assert progress.current_phase == "main"
        assert progress.primary_phase == "main"
        assert progress.is_complete is False


class TestNamedPhaseStatus:
    """The operator stamps names into CR status and kinds into phase details."""

    def test_annotations_prefer_named_phase_over_legacy_aggregate(self) -> None:
        annotations = _build_progress_annotations(
            {
                "main": _stats("main", "profiling", start_ns=10),
                "profiling": CombinedPhaseStats(
                    phase=CreditPhase.PROFILING,
                    start_ns=100,
                    requests_completed=50,
                    total_expected_requests=50,
                ),
            },
            SystemState.PROFILING,
        )

        assert annotations[ProgressAnnotations.PHASE] == "main"
