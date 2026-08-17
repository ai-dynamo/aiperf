# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A recorded fatal control-node error must break BOTH phase waits.

The runner has two sequential waits. Setting only ``all_credits_returned_event``
leaves a phase parked in ``_wait_for_sending_complete`` blocked forever when the
run has no explicit duration (``time_left_in_seconds()`` is None, so the wait has
no timeout) -- with the recorded error sitting unread.
"""

from __future__ import annotations

import pytest


def _tracker():
    from unittest.mock import MagicMock

    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker

    return PhaseProgressTracker(MagicMock())


def test_fatal_error_sets_sent_event() -> None:
    tracker = _tracker()
    assert not tracker.all_credits_sent_event.is_set()

    tracker.record_fatal_error(RuntimeError("control node exploded"))

    assert tracker.all_credits_sent_event.is_set(), (
        "a phase parked in _wait_for_sending_complete would hang forever"
    )


def test_fatal_error_sets_returned_event() -> None:
    tracker = _tracker()
    tracker.record_fatal_error(RuntimeError("boom"))
    assert tracker.all_credits_returned_event.is_set()


def test_fatal_error_is_recorded_and_retrievable() -> None:
    tracker = _tracker()
    error = RuntimeError("boom")
    tracker.record_fatal_error(error)
    assert tracker.fatal_error is error


def test_first_error_wins() -> None:
    """A later error must not mask the original cause."""
    tracker = _tracker()
    first = RuntimeError("first")
    tracker.record_fatal_error(first)
    tracker.record_fatal_error(RuntimeError("second"))
    assert tracker.fatal_error is first


@pytest.mark.asyncio
async def test_both_waits_return_promptly_after_fatal_error() -> None:
    """Neither event wait blocks once a fatal error is recorded."""
    import asyncio

    tracker = _tracker()
    tracker.record_fatal_error(RuntimeError("boom"))

    await asyncio.wait_for(tracker.all_credits_sent_event.wait(), timeout=1)
    await asyncio.wait_for(tracker.all_credits_returned_event.wait(), timeout=1)
