# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.models.credit_models import (
    BasePhaseStats,
    CreditPhaseStats,
    PhaseRecordsStats,
    ProcessingStats,
)


@pytest.fixture
def base_phase_stats():
    """Factory for BasePhaseStats instances."""

    def _create(**kwargs) -> BasePhaseStats:
        defaults = {"phase": CreditPhase.WARMUP}
        defaults.update(kwargs)
        return BasePhaseStats(**defaults)

    return _create


@pytest.fixture
def credit_phase_stats():
    """Factory for CreditPhaseStats instances."""

    def _create(**kwargs) -> CreditPhaseStats:
        defaults = {"phase": CreditPhase.PROFILING}
        defaults.update(kwargs)
        return CreditPhaseStats(**defaults)

    return _create


@pytest.fixture
def phase_records_stats():
    """Factory for PhaseRecordsStats instances."""

    def _create(**kwargs) -> PhaseRecordsStats:
        defaults = {"phase": CreditPhase.PROFILING}
        defaults.update(kwargs)
        return PhaseRecordsStats(**defaults)

    return _create


@pytest.fixture
def processing_stats():
    """Factory for ProcessingStats instances."""

    def _create(**kwargs) -> ProcessingStats:
        return ProcessingStats(**kwargs)

    return _create


class TestBasePhaseStatsProperties:
    """Test BasePhaseStats property methods."""

    @pytest.mark.parametrize(
        "start_ns,expected",
        [
            (None, False),
            (0, True),
            (1000000, True),
            (1234567890123456789, True),
        ],  # fmt: skip
    )
    def test_is_started(self, base_phase_stats, start_ns, expected) -> None:
        stats = base_phase_stats(start_ns=start_ns)
        assert stats.is_started == expected

    @pytest.mark.parametrize(
        "sent_end_ns,expected",
        [
            (None, False),
            (0, True),
            (1000000, True),
            (1234567890123456789, True),
        ],  # fmt: skip
    )
    def test_is_sending_complete(self, base_phase_stats, sent_end_ns, expected) -> None:
        stats = base_phase_stats(sent_end_ns=sent_end_ns)
        assert stats.is_sending_complete == expected

    @pytest.mark.parametrize(
        "requests_end_ns,expected",
        [
            (None, False),
            (0, True),
            (1000000, True),
            (1234567890123456789, True),
        ],  # fmt: skip
    )
    def test_is_requests_complete(
        self, base_phase_stats, requests_end_ns, expected
    ) -> None:
        stats = base_phase_stats(requests_end_ns=requests_end_ns)
        assert stats.is_requests_complete == expected


class TestBasePhaseStatsValidation:
    """Test BasePhaseStats field validation."""

    def test_requires_phase(self) -> None:
        with pytest.raises(ValueError, match="Field required"):
            BasePhaseStats()

    @pytest.mark.parametrize("field", ["start_ns", "sent_end_ns", "requests_end_ns"])
    def test_timestamp_fields_must_be_non_negative(
        self, base_phase_stats, field
    ) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            base_phase_stats(**{field: -1})

    @pytest.mark.parametrize(
        "field",
        ["total_expected_requests", "expected_duration_sec", "expected_num_sessions"],
    )
    def test_expected_fields_must_be_positive(self, base_phase_stats, field) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            base_phase_stats(**{field: 0})

    @pytest.mark.parametrize(
        "field",
        [
            "final_requests_sent",
            "final_requests_completed",
            "final_requests_cancelled",
            "final_request_errors",
            "final_sent_sessions",
            "final_completed_sessions",
            "final_cancelled_sessions",
        ],
    )
    def test_final_count_fields_must_be_non_negative(
        self, base_phase_stats, field
    ) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            base_phase_stats(**{field: -1})


class TestCreditPhaseStatsProperties:
    """Test CreditPhaseStats calculated properties."""

    @pytest.mark.parametrize(
        "sent,completed,cancelled,expected",
        [
            (0, 0, 0, 0),
            (10, 0, 0, 10),
            (10, 5, 0, 5),
            (10, 5, 2, 3),
            (100, 80, 15, 5),
        ],  # fmt: skip
    )
    def test_in_flight_sessions(
        self, credit_phase_stats, sent, completed, cancelled, expected
    ) -> None:
        stats = credit_phase_stats(
            sent_sessions=sent,
            completed_sessions=completed,
            cancelled_sessions=cancelled,
        )
        assert stats.in_flight_sessions == expected

    @pytest.mark.parametrize(
        "sent,completed,cancelled,expected",
        [
            (0, 0, 0, 0),
            (10, 0, 0, 10),
            (10, 5, 0, 5),
            (10, 5, 2, 3),
            (100, 80, 15, 5),
        ],  # fmt: skip
    )
    def test_in_flight_requests(
        self, credit_phase_stats, sent, completed, cancelled, expected
    ) -> None:
        stats = credit_phase_stats(
            requests_sent=sent,
            requests_completed=completed,
            requests_cancelled=cancelled,
        )
        assert stats.in_flight_requests == expected


class TestCreditPhaseStatsRequestsElapsedTime:
    """Test requests_elapsed_time property."""

    def test_returns_zero_when_not_started(self, credit_phase_stats) -> None:
        stats = credit_phase_stats(start_ns=None)
        assert stats.requests_elapsed_time == 0.0

    def test_returns_elapsed_time_when_completed(self, credit_phase_stats) -> None:
        start_ns = 1000000000
        end_ns = start_ns + (5 * NANOS_PER_SECOND)
        stats = credit_phase_stats(start_ns=start_ns, requests_end_ns=end_ns)
        assert stats.requests_elapsed_time == 5.0

    def test_returns_current_elapsed_time_when_in_progress(
        self, credit_phase_stats, time_traveler
    ) -> None:
        start_ns = time.time_ns()
        stats = credit_phase_stats(start_ns=start_ns)
        time_traveler.advance_time(3.0)
        assert stats.requests_elapsed_time == pytest.approx(3.0, abs=0.05)


class TestCreditPhaseStatsRequestsErrorPercent:
    """Test requests_error_percent property."""

    def test_returns_zero_when_no_requests_completed(self, credit_phase_stats) -> None:
        stats = credit_phase_stats(requests_completed=0, request_errors=0)
        assert stats.requests_error_percent == 0.0

    def test_uses_final_values_when_available(self, credit_phase_stats) -> None:
        stats = credit_phase_stats(
            final_requests_completed=100,
            final_request_errors=25,
            requests_completed=50,
            request_errors=10,
        )
        assert stats.requests_error_percent == 25.0

    @pytest.mark.parametrize(
        "completed,errors,expected_pct",
        [
            (100, 0, 0.0),
            (100, 10, 10.0),
            (100, 25, 25.0),
            (100, 100, 100.0),
            (50, 5, 10.0),
        ],  # fmt: skip
    )
    def test_calculates_error_percent_from_current_values(
        self, credit_phase_stats, completed, errors, expected_pct
    ) -> None:
        stats = credit_phase_stats(requests_completed=completed, request_errors=errors)
        assert stats.requests_error_percent == expected_pct


class TestCreditPhaseStatsRequestsProgressPercent:
    """Test requests_progress_percent property."""

    def test_returns_none_when_not_started(self, credit_phase_stats) -> None:
        stats = credit_phase_stats(start_ns=None)
        assert stats.requests_progress_percent is None

    def test_returns_100_when_requests_complete(self, credit_phase_stats) -> None:
        stats = credit_phase_stats(start_ns=1000, requests_end_ns=2000)
        assert stats.requests_progress_percent == 100

    def test_returns_none_when_no_expectations_set(
        self, credit_phase_stats, time_traveler
    ) -> None:
        stats = credit_phase_stats(
            start_ns=time.time_ns(),
            total_expected_requests=None,
            expected_duration_sec=None,
            expected_num_sessions=None,
        )
        assert stats.requests_progress_percent is None

    def test_calculates_from_request_count(
        self, credit_phase_stats, time_traveler
    ) -> None:
        stats = credit_phase_stats(
            start_ns=time.time_ns(),
            total_expected_requests=100,
            requests_completed=50,
        )
        assert stats.requests_progress_percent == 50.0

    def test_calculates_from_time_elapsed(
        self, credit_phase_stats, time_traveler
    ) -> None:
        start_ns = time.time_ns()
        stats = credit_phase_stats(start_ns=start_ns, expected_duration_sec=10.0)
        time_traveler.advance_time(5.0)
        assert stats.requests_progress_percent == pytest.approx(50.0, abs=2.0)

    def test_calculates_from_session_count(
        self, credit_phase_stats, time_traveler
    ) -> None:
        stats = credit_phase_stats(
            start_ns=time.time_ns(),
            expected_num_sessions=200,
            completed_sessions=100,
        )
        assert stats.requests_progress_percent == 50.0

    def test_returns_max_when_multiple_conditions(
        self, credit_phase_stats, time_traveler
    ) -> None:
        start_ns = time.time_ns()
        stats = credit_phase_stats(
            start_ns=start_ns,
            total_expected_requests=100,
            requests_completed=80,
            expected_duration_sec=10.0,
            expected_num_sessions=100,
            completed_sessions=60,
        )
        time_traveler.advance_time(5.0)
        assert stats.requests_progress_percent == pytest.approx(80.0, abs=2.0)

    def test_caps_progress_at_100_percent(
        self, credit_phase_stats, time_traveler
    ) -> None:
        start_ns = time.time_ns()
        stats = credit_phase_stats(
            start_ns=start_ns,
            total_expected_requests=100,
            requests_completed=150,
        )
        assert stats.requests_progress_percent == 100.0


class TestCreditPhaseStatsValidation:
    """Test CreditPhaseStats field validation."""

    @pytest.mark.parametrize(
        "field",
        [
            "requests_sent",
            "requests_completed",
            "requests_cancelled",
            "request_errors",
            "sent_sessions",
            "completed_sessions",
            "cancelled_sessions",
            "total_session_turns",
        ],
    )
    def test_count_fields_must_be_non_negative(self, credit_phase_stats, field) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            credit_phase_stats(**{field: -1})


class TestPhaseRecordsStatsProperties:
    """Test PhaseRecordsStats property methods."""

    @pytest.mark.parametrize(
        "success,errors,expected",
        [(0, 0, 0), (10, 0, 10), (0, 5, 5), (100, 25, 125)],  # fmt: skip
    )
    def test_total_records(
        self, phase_records_stats, success, errors, expected
    ) -> None:
        stats = phase_records_stats(success_records=success, error_records=errors)
        assert stats.total_records == expected

    @pytest.mark.parametrize(
        "records_end_ns,expected",
        [(None, False), (0, True), (1000000, True)],  # fmt: skip
    )
    def test_is_records_complete(
        self, phase_records_stats, records_end_ns, expected
    ) -> None:
        stats = phase_records_stats(records_end_ns=records_end_ns)
        assert stats.is_records_complete == expected


class TestPhaseRecordsStatsRecordsElapsedTime:
    """Test records_elapsed_time property."""

    def test_returns_zero_when_not_started(self, phase_records_stats) -> None:
        stats = phase_records_stats(start_ns=None)
        assert stats.records_elapsed_time == 0.0

    def test_returns_elapsed_time_when_completed(self, phase_records_stats) -> None:
        start_ns = 1000000000
        end_ns = start_ns + (7 * NANOS_PER_SECOND)
        stats = phase_records_stats(start_ns=start_ns, records_end_ns=end_ns)
        assert stats.records_elapsed_time == 7.0

    def test_returns_current_elapsed_time_when_in_progress(
        self, phase_records_stats, time_traveler
    ) -> None:
        start_ns = time.time_ns()
        stats = phase_records_stats(start_ns=start_ns)
        time_traveler.advance_time(4.0)
        assert stats.records_elapsed_time == pytest.approx(4.0, abs=0.05)


class TestPhaseRecordsStatsRecordsErrorPercent:
    """Test records_error_percent property."""

    def test_returns_zero_when_no_records(self, phase_records_stats) -> None:
        stats = phase_records_stats(success_records=0, error_records=0)
        assert stats.records_error_percent == 0.0

    @pytest.mark.parametrize(
        "success,errors,expected_pct",
        [
            (100, 0, 0.0),
            (90, 10, 10.0),
            (75, 25, 25.0),
            (0, 100, 100.0),
            (50, 5, 9.090909090909092),
        ],  # fmt: skip
    )
    def test_calculates_error_percent(
        self, phase_records_stats, success, errors, expected_pct
    ) -> None:
        stats = phase_records_stats(success_records=success, error_records=errors)
        assert stats.records_error_percent == pytest.approx(expected_pct, abs=0.001)


class TestPhaseRecordsStatsRecordsProgressPercent:
    """Test records_progress_percent property."""

    def test_uses_final_requests_completed_when_available(
        self, phase_records_stats
    ) -> None:
        stats = phase_records_stats(
            success_records=80, error_records=20, final_requests_completed=200
        )
        assert stats.records_progress_percent == 50.0

    def test_uses_total_expected_requests_when_final_not_available(
        self, phase_records_stats
    ) -> None:
        stats = phase_records_stats(
            success_records=75, error_records=25, total_expected_requests=200
        )
        assert stats.records_progress_percent == 50.0

    def test_returns_none_when_no_expectations(self, phase_records_stats) -> None:
        stats = phase_records_stats(success_records=50, error_records=10)
        assert stats.records_progress_percent is None


class TestPhaseRecordsStatsValidation:
    """Test PhaseRecordsStats field validation."""

    @pytest.mark.parametrize("field", ["success_records", "error_records"])
    def test_record_fields_must_be_non_negative(
        self, phase_records_stats, field
    ) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            phase_records_stats(**{field: -1})

    def test_records_end_ns_must_be_non_negative(self, phase_records_stats) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            phase_records_stats(records_end_ns=-1)


class TestProcessingStatsProperties:
    """Test ProcessingStats property methods."""

    @pytest.mark.parametrize(
        "processed,errors,expected",
        [
            (0, 0, 0),
            (100, 0, 100),
            (0, 50, 50),
            (100, 25, 125),
            (500, 100, 600),
        ],  # fmt: skip
    )
    def test_total_records(self, processing_stats, processed, errors, expected) -> None:
        stats = processing_stats(processed=processed, errors=errors)
        assert stats.total_records == expected


class TestProcessingStatsValidation:
    """Test ProcessingStats field validation."""

    def test_defaults_to_zero(self, processing_stats) -> None:
        stats = processing_stats()
        assert stats.processed == 0
        assert stats.errors == 0


class TestCreditPhaseStatsIntegration:
    """Integration tests for CreditPhaseStats."""

    def test_warmup_phase_lifecycle(self, credit_phase_stats, time_traveler) -> None:
        start_ns = time.time_ns()

        stats = credit_phase_stats(
            phase=CreditPhase.WARMUP, start_ns=start_ns, total_expected_requests=100
        )
        assert stats.is_started
        assert not stats.is_sending_complete
        assert not stats.is_requests_complete

        stats = credit_phase_stats(
            phase=CreditPhase.WARMUP,
            start_ns=start_ns,
            total_expected_requests=100,
            requests_sent=50,
            requests_completed=30,
        )
        assert stats.in_flight_requests == 20
        assert stats.requests_progress_percent == 30.0

        stats = credit_phase_stats(
            phase=CreditPhase.WARMUP,
            start_ns=start_ns,
            sent_end_ns=start_ns + (5 * NANOS_PER_SECOND),
            total_expected_requests=100,
            requests_sent=100,
            requests_completed=100,
        )
        assert stats.is_sending_complete
        assert stats.in_flight_requests == 0

        stats = credit_phase_stats(
            phase=CreditPhase.WARMUP,
            start_ns=start_ns,
            sent_end_ns=start_ns + (5 * NANOS_PER_SECOND),
            requests_end_ns=start_ns + (10 * NANOS_PER_SECOND),
            total_expected_requests=100,
            requests_sent=100,
            requests_completed=100,
        )
        assert stats.is_requests_complete
        assert stats.requests_elapsed_time == 10.0


class TestPhaseRecordsStatsIntegration:
    """Integration tests for PhaseRecordsStats."""

    def test_records_processing_lifecycle(
        self, phase_records_stats, time_traveler
    ) -> None:
        start_ns = time.time_ns()

        stats = phase_records_stats(
            phase=CreditPhase.PROFILING,
            start_ns=start_ns,
            final_requests_completed=100,
            success_records=60,
            error_records=10,
        )
        assert stats.total_records == 70
        assert stats.records_progress_percent == 70.0
        assert stats.records_error_percent == pytest.approx(14.285714, abs=0.001)
        assert not stats.is_records_complete

        time_traveler.advance_time(5.0)
        end_ns = time.time_ns()
        stats = phase_records_stats(
            phase=CreditPhase.PROFILING,
            start_ns=start_ns,
            records_end_ns=end_ns,
            final_requests_completed=100,
            success_records=95,
            error_records=5,
        )
        assert stats.is_records_complete
        assert stats.records_elapsed_time == pytest.approx(5.0, abs=0.05)
        assert stats.total_records == 100
        assert stats.records_progress_percent == 100.0
        assert stats.records_error_percent == 5.0
