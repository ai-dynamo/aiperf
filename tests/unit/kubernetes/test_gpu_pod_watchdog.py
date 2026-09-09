# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-fast contracts for GPU E2E pod readiness."""

from __future__ import annotations

from tests.kubernetes.helpers.kubectl import PodStatus
from tests.kubernetes.helpers.pod_watchdog import detect_fatal_image_conditions


def test_detect_fatal_image_conditions_crash_loop_reports_container_failure() -> None:
    """CrashLoopBackOff must not consume the full GPU readiness timeout."""
    pods = [
        PodStatus(
            name="trtllm-server-0",
            namespace="trtllm-server",
            phase="Running",
            ready="0/1",
            restarts=5,
            containers={
                "trtllm": {
                    "state": {
                        "waiting": {
                            "reason": "CrashLoopBackOff",
                            "message": "back-off 5m0s restarting failed container",
                        }
                    }
                }
            },
        )
    ]

    message = detect_fatal_image_conditions(pods, "trtllm-server")

    assert message is not None
    assert "CrashLoopBackOff" in message
    assert "trtllm" in message
