# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for records_tracker.

Focuses on:
- CreditPhaseRecordsTracker counter / timestamp / completion-latch behavior
- RecordsTracker per-phase routing, results-window aggregation, and
  cross-phase worker-stats roll-up
"""

import time

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.metric_records_wire import (
    MetricRecordMetadata,
    MetricRecordsData,
    WireErrorDetails,
)
from aiperf.common.models import (
    CreditPhaseStats,
    PhaseRecordsStats,
    WorkerProcessingStats,
)
from aiperf.records.records_tracker import (
    CreditPhaseRecordsTracker,
    RecordsTracker,
)

PROFILING: CreditPhase = "profiling"
WARMUP: CreditPhase = "warmup"
COOLDOWN: CreditPhase = "cooldown"


# ============================================================
# Helpers
# ============================================================


def make_record(
    *,
    phase: CreditPhase = PROFILING,
    worker_id: str = "worker-1",
    valid: bool = True,
) -> MetricRecordsData:
    """Build a MetricRecordsData with the minimal fields the tracker reads."""
    error: WireErrorDetails | None = None if valid else WireErrorDetails(message="boom")
    return MetricRecordsData(
        metadata=MetricRecordMetadata(
            session_num=0,
            request_start_ns=1,
            request_end_ns=2,
            worker_id=worker_id,
            record_processor_id="rp-1",
            benchmark_phase=phase,
        ),
        metrics={},
        error=error,
    )


def make_credit_stats(
    *,
    phase: CreditPhase = PROFILING,
    exclude_from_results: bool = False,
    start_ns: int | None = 1_000,
    sent_end_ns: int | None = 2_000,
    requests_end_ns: int | None = 3_000,
    total_expected_requests: int | None = 100,
    final_requests_completed: int | None = 100,
    final_requests_cancelled: int | None = 0,
    final_request_errors: int | None = 0,
    timeout_triggered: bool = False,
    grace_period_timeout_triggered: bool = False,
    was_cancelled: bool = False,
) -> CreditPhaseStats:
    """Build a CreditPhaseStats for update_from_credit_phase_stats tests."""
    return CreditPhaseStats(
        phase=phase,
        exclude_from_results=exclude_from_results,
        start_ns=start_ns,
        sent_end_ns=sent_end_ns,
        requests_end_ns=requests_end_ns,
        total_expected_requests=total_expected_requests,
        final_requests_completed=final_requests_completed,
        final_requests_cancelled=final_requests_cancelled,
        final_request_errors=final_request_errors,
        timeout_triggered=timeout_triggered,
        grace_period_timeout_triggered=grace_period_timeout_triggered,
        was_cancelled=was_cancelled,
    )


# ============================================================
# CreditPhaseRecordsTracker
# ============================================================


class TestCreditPhaseRecordsTrackerConstruction:
    """Verify default state at construction."""

    def test_init_zeros_all_counters_and_timestamps(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)

        assert tracker.phase == PROFILING
        assert tracker.total_records == 0
        assert tracker.is_active is False
        assert tracker._success_records == 0
        assert tracker._error_records == 0
        assert tracker._start_ns is None
        assert tracker._sent_end_ns is None
        assert tracker._requests_end_ns is None
        assert tracker._records_end_ns is None
        assert tracker._final_requests_completed is None
        assert tracker._final_requests_cancelled is None
        assert tracker._final_request_errors is None
        assert tracker._timeout_triggered is False
        assert tracker._grace_period_timeout_triggered is False
        assert tracker._was_cancelled is False
        assert tracker._sent_all_records_received is False
        # None (not False) until CreditPhaseStats registers the phase — the
        # tri-state lets racing records buffer instead of leaking into results.
        assert tracker._exclude_from_results is None
        assert tracker._total_expected_requests is None
        assert tracker._worker_stats == {}


class TestCreditPhaseRecordsTrackerCounters:
    """Verify increment_* methods and total_records aggregation."""

    def test_increment_success_records_advances_counter(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker.increment_success_records()
        tracker.increment_success_records()
        tracker.increment_success_records()

        assert tracker._success_records == 3
        assert tracker._error_records == 0
        assert tracker.total_records == 3

    def test_increment_error_records_advances_counter(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker.increment_error_records()
        tracker.increment_error_records()

        assert tracker._error_records == 2
        assert tracker._success_records == 0
        assert tracker.total_records == 2

    def test_total_records_sums_success_and_error(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        for _ in range(4):
            tracker.increment_success_records()
        for _ in range(3):
            tracker.increment_error_records()

        assert tracker.total_records == 7

    def test_increment_worker_success_creates_per_worker_stats(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker.increment_worker_success_records("worker-a")
        tracker.increment_worker_success_records("worker-a")
        tracker.increment_worker_success_records("worker-b")

        assert tracker._worker_stats["worker-a"].success_records == 2
        assert tracker._worker_stats["worker-b"].success_records == 1
        assert tracker._worker_stats["worker-a"].error_records == 0

    def test_increment_worker_error_creates_per_worker_stats(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker.increment_worker_error_records("worker-a")
        tracker.increment_worker_error_records("worker-b")
        tracker.increment_worker_error_records("worker-b")

        assert tracker._worker_stats["worker-a"].error_records == 1
        assert tracker._worker_stats["worker-b"].error_records == 2

    def test_first_worker_id_auto_creates_worker_processing_stats(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker.increment_worker_success_records("first-time-worker")

        assert isinstance(
            tracker._worker_stats["first-time-worker"], WorkerProcessingStats
        )
        assert tracker._worker_stats["first-time-worker"].success_records == 1
        assert tracker._worker_stats["first-time-worker"].error_records == 0


class TestCreditPhaseRecordsTrackerUpdateFromCreditPhaseStats:
    """Verify all fields copied from CreditPhaseStats onto the tracker."""

    def test_update_from_credit_phase_stats_copies_all_eleven_fields(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        credit_stats = make_credit_stats(
            phase=PROFILING,
            exclude_from_results=True,
            start_ns=1_111,
            sent_end_ns=2_222,
            requests_end_ns=3_333,
            total_expected_requests=42,
            final_requests_completed=40,
            final_requests_cancelled=1,
            final_request_errors=1,
            timeout_triggered=True,
            grace_period_timeout_triggered=True,
            was_cancelled=True,
        )

        tracker.update_from_credit_phase_stats(credit_stats)

        assert tracker._exclude_from_results is True
        assert tracker._start_ns == 1_111
        assert tracker._sent_end_ns == 2_222
        assert tracker._requests_end_ns == 3_333
        assert tracker._total_expected_requests == 42
        assert tracker._final_requests_completed == 40
        assert tracker._final_requests_cancelled == 1
        assert tracker._final_request_errors == 1
        assert tracker._timeout_triggered is True
        assert tracker._grace_period_timeout_triggered is True
        assert tracker._was_cancelled is True


class TestCreditPhaseRecordsTrackerIsActive:
    """is_active = start_ns set AND records_end_ns unset."""

    def test_is_active_false_before_start(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        assert tracker.is_active is False

    def test_is_active_true_after_start_before_records_end(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker._start_ns = 100
        assert tracker.is_active is True

    def test_is_active_false_after_records_end_set(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker._start_ns = 100
        tracker._records_end_ns = 200
        assert tracker.is_active is False


class TestCreditPhaseRecordsTrackerCreateStats:
    """create_stats produces a populated, fresh PhaseRecordsStats each call."""

    def test_create_stats_populates_all_fields(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker.update_from_credit_phase_stats(
            make_credit_stats(
                phase=PROFILING,
                exclude_from_results=True,
                start_ns=10,
                sent_end_ns=20,
                requests_end_ns=30,
                total_expected_requests=5,
                final_requests_completed=5,
                final_requests_cancelled=0,
                final_request_errors=0,
                timeout_triggered=True,
                grace_period_timeout_triggered=False,
                was_cancelled=False,
            )
        )
        tracker.increment_success_records()
        tracker.increment_error_records()
        tracker._records_end_ns = 99

        stats = tracker.create_stats()

        assert isinstance(stats, PhaseRecordsStats)
        assert stats.phase == PROFILING
        assert stats.exclude_from_results is True
        assert stats.start_ns == 10
        assert stats.sent_end_ns == 20
        assert stats.requests_end_ns == 30
        assert stats.records_end_ns == 99
        assert stats.total_expected_requests == 5
        assert stats.success_records == 1
        assert stats.error_records == 1
        assert stats.final_requests_completed == 5
        assert stats.final_requests_cancelled == 0
        assert stats.final_request_errors == 0
        assert stats.timeout_triggered is True
        assert stats.grace_period_timeout_triggered is False
        assert stats.was_cancelled is False

    def test_create_stats_returns_fresh_instance_each_call(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        first = tracker.create_stats()
        second = tracker.create_stats()

        assert first is not second


class TestCreditPhaseRecordsTrackerCheckAndSetAllRecordsReceived:
    """Verify the latching all-records-received signal."""

    def test_returns_false_when_final_requests_completed_is_none(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker.increment_success_records()
        tracker.increment_success_records()

        assert tracker.check_and_set_all_records_received() is False
        assert tracker._records_end_ns is None
        assert tracker._sent_all_records_received is False

    def test_returns_false_when_total_records_below_threshold(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker._final_requests_completed = 5
        tracker.increment_success_records()
        tracker.increment_error_records()

        assert tracker.check_and_set_all_records_received() is False
        assert tracker._records_end_ns is None
        assert tracker._sent_all_records_received is False

    @pytest.mark.parametrize(
        "successes,errors,target",
        [
            (5, 0, 5),
            (3, 2, 5),
            param(10, 0, 5, id="exceeds-target"),
        ],
    )  # fmt: skip
    def test_returns_true_and_latches_when_threshold_reached(
        self, successes: int, errors: int, target: int
    ) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker._final_requests_completed = target
        for _ in range(successes):
            tracker.increment_success_records()
        for _ in range(errors):
            tracker.increment_error_records()

        before = time.time_ns()
        result = tracker.check_and_set_all_records_received()
        after = time.time_ns()

        assert result is True
        assert tracker._sent_all_records_received is True
        assert tracker._records_end_ns is not None
        assert before <= tracker._records_end_ns <= after

    def test_subsequent_calls_return_false_after_first_latch(self) -> None:
        tracker = CreditPhaseRecordsTracker(PROFILING)
        tracker._final_requests_completed = 2
        tracker.increment_success_records()
        tracker.increment_success_records()

        assert tracker.check_and_set_all_records_received() is True
        first_end = tracker._records_end_ns

        # Even with more records, latched flag means subsequent calls return False
        tracker.increment_success_records()
        assert tracker.check_and_set_all_records_received() is False
        # records_end_ns is not overwritten
        assert tracker._records_end_ns == first_end


# ============================================================
# RecordsTracker
# ============================================================


class TestRecordsTrackerPhaseLookup:
    """_get_phase_tracker is lazy + memoized."""

    def test_get_phase_tracker_lazily_creates_for_unknown_phase(self) -> None:
        rt = RecordsTracker()
        tracker = rt._get_phase_tracker(PROFILING)

        assert isinstance(tracker, CreditPhaseRecordsTracker)
        assert tracker.phase == PROFILING

    def test_get_phase_tracker_returns_same_instance_on_repeat_call(self) -> None:
        rt = RecordsTracker()
        first = rt._get_phase_tracker(PROFILING)
        second = rt._get_phase_tracker(PROFILING)

        assert first is second

    def test_distinct_phases_get_distinct_trackers(self) -> None:
        rt = RecordsTracker()
        warmup = rt._get_phase_tracker(WARMUP)
        profiling = rt._get_phase_tracker(PROFILING)

        assert warmup is not profiling
        assert warmup.phase == WARMUP
        assert profiling.phase == PROFILING


class TestRecordsTrackerUpdateFromRecordData:
    """Verify routing of valid/invalid records to per-phase counters."""

    def test_valid_record_increments_success_counters(self) -> None:
        rt = RecordsTracker()
        rt.update_from_record_data(
            make_record(phase=PROFILING, worker_id="worker-1", valid=True)
        )

        phase = rt._get_phase_tracker(PROFILING)
        assert phase._success_records == 1
        assert phase._error_records == 0
        assert phase._worker_stats["worker-1"].success_records == 1
        assert phase._worker_stats["worker-1"].error_records == 0

    def test_invalid_record_increments_error_counters(self) -> None:
        rt = RecordsTracker()
        rt.update_from_record_data(
            make_record(phase=PROFILING, worker_id="worker-1", valid=False)
        )

        phase = rt._get_phase_tracker(PROFILING)
        assert phase._error_records == 1
        assert phase._success_records == 0
        assert phase._worker_stats["worker-1"].error_records == 1
        assert phase._worker_stats["worker-1"].success_records == 0

    def test_record_data_routes_to_correct_phase_tracker(self) -> None:
        rt = RecordsTracker()
        rt.update_from_record_data(make_record(phase=WARMUP, worker_id="w1"))
        rt.update_from_record_data(make_record(phase=PROFILING, worker_id="w1"))
        rt.update_from_record_data(make_record(phase=PROFILING, worker_id="w1"))

        assert rt._get_phase_tracker(WARMUP)._success_records == 1
        assert rt._get_phase_tracker(PROFILING)._success_records == 2


class TestRecordsTrackerPhaseInfo:
    """update_phase_info / is_phase_excluded / cancellation flags."""

    def test_update_phase_info_forwards_to_correct_phase(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(
            make_credit_stats(phase=PROFILING, total_expected_requests=99)
        )

        phase = rt._get_phase_tracker(PROFILING)
        assert phase._total_expected_requests == 99

    def test_is_phase_excluded_default_is_false(self) -> None:
        rt = RecordsTracker()
        assert rt.is_phase_excluded(PROFILING) is False

    def test_is_phase_excluded_reflects_credit_stats_value(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(make_credit_stats(phase=WARMUP, exclude_from_results=True))
        assert rt.is_phase_excluded(WARMUP) is True
        assert rt.is_phase_excluded(PROFILING) is False

    def test_get_phase_exclusion_is_none_until_stats_register(self) -> None:
        """Tri-state gate: unknown phases must be distinguishable from known
        included phases so racing records can be buffered, not leaked."""
        rt = RecordsTracker()
        assert rt.get_phase_exclusion(PROFILING) is None
        # A record arriving first creates the tracker but does NOT classify it.
        rt.update_from_record_data(make_record(phase=PROFILING))
        assert rt.get_phase_exclusion(PROFILING) is None

        rt.update_phase_info(make_credit_stats(phase=PROFILING))
        assert rt.get_phase_exclusion(PROFILING) is False
        rt.update_phase_info(make_credit_stats(phase=WARMUP, exclude_from_results=True))
        assert rt.get_phase_exclusion(WARMUP) is True

    def test_get_phase_exclusion_probe_does_not_create_tracker(self) -> None:
        rt = RecordsTracker()
        assert rt.get_phase_exclusion(WARMUP) is None
        assert WARMUP not in rt._phase_trackers

    def test_was_phase_cancelled_default_is_false(self) -> None:
        rt = RecordsTracker()
        assert rt.was_phase_cancelled(PROFILING) is False

    def test_mark_phase_cancelled_sets_flag(self) -> None:
        rt = RecordsTracker()
        rt.mark_phase_cancelled(PROFILING)

        assert rt.was_phase_cancelled(PROFILING) is True
        assert rt.was_phase_cancelled(WARMUP) is False

    def test_check_and_set_all_records_received_for_phase_forwards(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(
            make_credit_stats(phase=PROFILING, final_requests_completed=2)
        )
        rt.update_from_record_data(make_record(phase=PROFILING))
        rt.update_from_record_data(make_record(phase=PROFILING))

        assert rt.check_and_set_all_records_received_for_phase(PROFILING) is True
        # Second call latched.
        assert rt.check_and_set_all_records_received_for_phase(PROFILING) is False


class TestRecordsTrackerActiveAndResultsPhases:
    """create_active_phase_stats_list + get_results_phases visibility filters."""

    def test_create_active_phase_stats_list_only_includes_active(self) -> None:
        rt = RecordsTracker()
        # Active: started, no records_end_ns.
        rt.update_phase_info(
            make_credit_stats(phase=PROFILING, start_ns=100, requests_end_ns=None)
        )
        # Inactive: never started.
        rt._get_phase_tracker(WARMUP)
        # Inactive: started + records_end_ns set manually.
        rt.update_phase_info(make_credit_stats(phase=COOLDOWN, start_ns=50))
        rt._get_phase_tracker(COOLDOWN)._records_end_ns = 500

        active = rt.create_active_phase_stats_list()

        assert len(active) == 1
        assert active[0].phase == PROFILING

    def test_create_active_phase_stats_list_returns_phase_records_stats(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(
            make_credit_stats(phase=PROFILING, start_ns=100, requests_end_ns=None)
        )

        active = rt.create_active_phase_stats_list()
        assert isinstance(active[0], PhaseRecordsStats)

    def test_get_results_phases_filters_excluded(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(make_credit_stats(phase=WARMUP, exclude_from_results=True))
        rt.update_phase_info(
            make_credit_stats(phase=PROFILING, exclude_from_results=False)
        )

        result = rt.get_results_phases()
        assert PROFILING in result
        assert WARMUP not in result

    def test_get_results_phases_empty_when_all_excluded(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(make_credit_stats(phase=WARMUP, exclude_from_results=True))
        rt.update_phase_info(
            make_credit_stats(phase=COOLDOWN, exclude_from_results=True)
        )

        assert rt.get_results_phases() == []


class TestRecordsTrackerTimeWindow:
    """get_results_time_window behavior across excluded/missing phases."""

    def test_returns_none_none_when_no_phases(self) -> None:
        rt = RecordsTracker()
        assert rt.get_results_time_window() == (None, None)

    def test_returns_min_start_max_end_across_non_excluded(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(
            make_credit_stats(
                phase=PROFILING,
                start_ns=100,
                requests_end_ns=500,
                exclude_from_results=False,
            )
        )
        rt.update_phase_info(
            make_credit_stats(
                phase=COOLDOWN,
                start_ns=50,
                requests_end_ns=900,
                exclude_from_results=False,
            )
        )

        start, end = rt.get_results_time_window()
        assert start == 50
        assert end == 900

    def test_skips_excluded_phases(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(
            make_credit_stats(
                phase=WARMUP,
                start_ns=10,
                requests_end_ns=9_999,
                exclude_from_results=True,
            )
        )
        rt.update_phase_info(
            make_credit_stats(
                phase=PROFILING,
                start_ns=100,
                requests_end_ns=500,
                exclude_from_results=False,
            )
        )

        start, end = rt.get_results_time_window()
        # Excluded WARMUP is ignored even though its start (10) is earlier
        # and its end (9_999) is later.
        assert start == 100
        assert end == 500

    def test_returns_none_none_when_all_phases_excluded(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(
            make_credit_stats(
                phase=WARMUP,
                start_ns=10,
                requests_end_ns=20,
                exclude_from_results=True,
            )
        )
        assert rt.get_results_time_window() == (None, None)

    def test_tolerates_missing_start_or_end(self) -> None:
        rt = RecordsTracker()
        # Phase with start but no end.
        rt.update_phase_info(
            make_credit_stats(
                phase=PROFILING,
                start_ns=100,
                requests_end_ns=None,
                exclude_from_results=False,
            )
        )
        # Phase with end but no start.
        rt.update_phase_info(
            make_credit_stats(
                phase=COOLDOWN,
                start_ns=None,
                requests_end_ns=999,
                exclude_from_results=False,
            )
        )

        start, end = rt.get_results_time_window()
        assert start == 100
        assert end == 999

    def test_returns_none_none_when_no_phase_has_either_timestamp(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(
            make_credit_stats(
                phase=PROFILING,
                start_ns=None,
                requests_end_ns=None,
                exclude_from_results=False,
            )
        )

        assert rt.get_results_time_window() == (None, None)


class TestRecordsTrackerAreAllResultsPhasesComplete:
    """Verify the are_all_results_phases_complete contract."""

    def test_returns_false_when_no_results_phases(self) -> None:
        rt = RecordsTracker()
        # Critical: empty results phases returns False, NOT True.
        assert rt.are_all_results_phases_complete() is False

    def test_returns_false_when_all_phases_excluded(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(make_credit_stats(phase=WARMUP, exclude_from_results=True))
        assert rt.are_all_results_phases_complete() is False

    def test_returns_false_when_some_phases_not_complete(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(
            make_credit_stats(
                phase=PROFILING,
                final_requests_completed=2,
                exclude_from_results=False,
            )
        )
        rt.update_phase_info(
            make_credit_stats(
                phase=COOLDOWN,
                final_requests_completed=1,
                exclude_from_results=False,
            )
        )
        # Complete profiling only.
        rt.update_from_record_data(make_record(phase=PROFILING))
        rt.update_from_record_data(make_record(phase=PROFILING))
        rt.check_and_set_all_records_received_for_phase(PROFILING)

        assert rt.are_all_results_phases_complete() is False

    def test_returns_true_when_every_results_phase_complete(self) -> None:
        rt = RecordsTracker()
        rt.update_phase_info(
            make_credit_stats(
                phase=PROFILING,
                final_requests_completed=1,
                exclude_from_results=False,
            )
        )
        rt.update_phase_info(
            make_credit_stats(
                phase=WARMUP,
                final_requests_completed=1,
                exclude_from_results=True,  # excluded — not required for complete
            )
        )
        rt.update_from_record_data(make_record(phase=PROFILING))
        rt.check_and_set_all_records_received_for_phase(PROFILING)

        assert rt.are_all_results_phases_complete() is True


class TestRecordsTrackerCreateOverallWorkerStats:
    """create_overall_worker_stats sums per-worker counts across all phases."""

    def test_aggregates_single_worker_across_phases(self) -> None:
        rt = RecordsTracker()
        rt.update_from_record_data(make_record(phase=WARMUP, worker_id="w1"))
        rt.update_from_record_data(make_record(phase=WARMUP, worker_id="w1"))
        rt.update_from_record_data(make_record(phase=PROFILING, worker_id="w1"))
        rt.update_from_record_data(
            make_record(phase=PROFILING, worker_id="w1", valid=False)
        )

        result = rt.create_overall_worker_stats()
        assert "w1" in result
        assert result["w1"].success_records == 3
        assert result["w1"].error_records == 1

    def test_aggregates_multiple_workers_across_phases(self) -> None:
        rt = RecordsTracker()
        rt.update_from_record_data(make_record(phase=WARMUP, worker_id="w1"))
        rt.update_from_record_data(make_record(phase=PROFILING, worker_id="w1"))
        rt.update_from_record_data(make_record(phase=WARMUP, worker_id="w2"))
        rt.update_from_record_data(
            make_record(phase=PROFILING, worker_id="w2", valid=False)
        )
        rt.update_from_record_data(
            make_record(phase=PROFILING, worker_id="w2", valid=False)
        )

        result = rt.create_overall_worker_stats()
        assert result["w1"].success_records == 2
        assert result["w1"].error_records == 0
        assert result["w2"].success_records == 1
        assert result["w2"].error_records == 2

    def test_returns_empty_dict_when_no_phases_have_workers(self) -> None:
        rt = RecordsTracker()
        assert rt.create_overall_worker_stats() == {}

    def test_returns_plain_dict_not_defaultdict(self) -> None:
        rt = RecordsTracker()
        rt.update_from_record_data(make_record(worker_id="w1"))

        result = rt.create_overall_worker_stats()
        # Caller should not be able to silently auto-create new keys.
        assert type(result) is dict
        with pytest.raises(KeyError):
            _ = result["never-seen-worker"]
