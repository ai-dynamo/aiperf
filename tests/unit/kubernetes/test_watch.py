# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for AIPerfJob condition polling output."""

from unittest.mock import MagicMock

from aiperf.kubernetes.watch import _log_condition_updates


def _condition(*, status: str, message: str) -> dict[str, str]:
    return {
        "type": "Ready",
        "status": status,
        "reason": "ProbeResult",
        "message": message,
        "lastTransitionTime": "2026-08-04T12:00:00Z",
    }


def test_log_condition_updates_reports_same_length_status_transition() -> None:
    """Kubernetes updates list-map conditions in place instead of appending."""
    logger = MagicMock()
    previous = _log_condition_updates(
        logger,
        [_condition(status="False", message="controller is starting")],
        {},
        1.0,
    )
    logger.reset_mock()

    current = _log_condition_updates(
        logger,
        [_condition(status="True", message="controller is ready")],
        previous,
        2.0,
    )

    logger.info.assert_called_once()
    assert "PASS" in logger.info.call_args.args[0]
    assert "controller is ready" in logger.info.call_args.args[0]
    assert current != previous


def test_log_condition_updates_does_not_repeat_unchanged_condition() -> None:
    """Polling the same condition payload does not duplicate CLI output."""
    logger = MagicMock()
    condition = _condition(status="True", message="controller is ready")
    previous = _log_condition_updates(logger, [condition], {}, 1.0)
    logger.reset_mock()

    current = _log_condition_updates(logger, [condition], previous, 2.0)

    logger.info.assert_not_called()
    assert current == previous
