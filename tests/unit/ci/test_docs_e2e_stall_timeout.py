# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The container a documented command runs in must carry a stall timeout.

AIPerf declares a phase stalled after 600s with requests in flight and none
completing. A reasoning model answering a non-streaming request legitimately
generates for longer than that on a CI runner: on 2026-11-06 three shards
died this way while vLLM logged ``Running: 1 reqs`` at a steady 54 tokens/s
for the whole window, and an earlier run "recovered" at 722s, which is simply
when the request filled the context and stopped.

``docs/environment-variables.md`` names raising the timeout as the remedy for
precisely this workload, so the harness sets it on the container rather than
leaving every guide to work around a default tuned for real deployments.
"""

from __future__ import annotations

import inspect


def test_the_container_sets_the_stall_timeout() -> None:
    from constants import RECORD_PROGRESS_STALL_TIMEOUT
    from test_runner import EndToEndTestRunner

    source = inspect.getsource(EndToEndTestRunner._build_aiperf_container)

    assert "AIPERF_RECORD_PROGRESS_STALL_TIMEOUT" in source, (
        "the container runs with AIPerf's 600s default, which aborts a "
        "legitimately long non-streaming reasoning generation"
    )
    assert "RECORD_PROGRESS_STALL_TIMEOUT" in source, (
        "the value must come from the constant, not be written inline"
    )
    assert RECORD_PROGRESS_STALL_TIMEOUT > 600, (
        "raising the timeout is the whole point; this value is not a raise"
    )


def test_a_stall_is_reported_before_the_command_is_killed() -> None:
    """A stall must surface as a stall, not as an opaque command timeout."""
    from constants import AIPERF_COMMAND_TIMEOUT, RECORD_PROGRESS_STALL_TIMEOUT

    assert RECORD_PROGRESS_STALL_TIMEOUT < AIPERF_COMMAND_TIMEOUT, (
        f"a stall takes {RECORD_PROGRESS_STALL_TIMEOUT}s to report but the "
        f"harness kills the command at {AIPERF_COMMAND_TIMEOUT}s, so the run "
        f"dies with exit 124 and no diagnosis"
    )
