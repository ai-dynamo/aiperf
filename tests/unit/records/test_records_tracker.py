# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import CreditPhase
from aiperf.common.models import CreditPhaseStats
from aiperf.records.records_tracker import RecordsTracker


def _credit_phase_stats(**kwargs) -> CreditPhaseStats:
    defaults = {"phase": CreditPhase.PROFILING}
    defaults.update(kwargs)
    return CreditPhaseStats(**defaults)


class TestRecordsTrackerFinalRequestsSent:
    """RecordsTracker must carry final_requests_sent through to PhaseRecordsStats
    so downstream reporting (ProfileResults, JSON export) can surface it."""

    def test_carries_final_requests_sent_through(self) -> None:
        tracker = RecordsTracker()
        tracker.update_phase_info(
            _credit_phase_stats(
                final_requests_sent=100,
                final_requests_completed=90,
                final_requests_cancelled=0,
            )
        )

        stats = tracker.create_stats_for_phase(CreditPhase.PROFILING)

        assert stats.final_requests_sent == 100
        assert stats.final_requests_abandoned == 10

    def test_defaults_to_none_when_never_updated(self) -> None:
        tracker = RecordsTracker()

        stats = tracker.create_stats_for_phase(CreditPhase.PROFILING)

        assert stats.final_requests_sent is None
        assert stats.final_requests_abandoned is None
