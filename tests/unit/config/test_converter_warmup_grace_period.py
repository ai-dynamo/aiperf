# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for warmup-grace-period gating in build_warmup.

Covers a former bug in ``aiperf.config.flags._converter_warmup``: setting
``--warmup-grace-period`` without ``--warmup-duration`` (e.g. with
``--warmup-request-count`` or ``--warmup-num-sessions``) silently emitted
``grace_period`` on a non-duration warmup phase, which v2 ``PhaseConfig``
then rejected with the cryptic
``Phase 'warmup': grace_period requires duration to be set``.

build_warmup now raises a clear, action-oriented error at convert-time so
the user sees which flag combination is incompatible.
"""

from __future__ import annotations

import pytest

from aiperf.config.flags._converter_warmup import build_warmup
from aiperf.config.flags.cli_config import CLIConfig


def _make_user(loadgen: CLIConfig) -> CLIConfig:
    endpoint = CLIConfig(url="http://localhost:8000/test", model_names=["test-model"])
    return CLIConfig(
        **endpoint.model_dump(exclude_unset=True),
        **loadgen.model_dump(exclude_unset=True),
    )


class TestWarmupGracePeriodRequiresDuration:
    """--warmup-grace-period without --warmup-duration must raise clearly."""

    def test_warmup_grace_period_with_request_count_raises(self):
        loadgen = CLIConfig(
            warmup_request_count=10,
            warmup_grace_period=5.0,
        )
        with pytest.raises(ValueError, match="--warmup-grace-period requires"):
            build_warmup(_make_user(loadgen))

    def test_warmup_grace_period_with_num_sessions_raises(self):
        loadgen = CLIConfig(
            warmup_num_sessions=5,
            warmup_grace_period=5.0,
        )
        with pytest.raises(ValueError, match="--warmup-duration"):
            build_warmup(_make_user(loadgen))

    def test_error_message_mentions_offending_flags(self):
        """Error guides the user toward both fixes (set duration / drop grace)."""
        loadgen = CLIConfig(
            warmup_request_count=10,
            warmup_grace_period=5.0,
        )
        with pytest.raises(ValueError) as exc:
            build_warmup(_make_user(loadgen))
        msg = str(exc.value)
        assert "--warmup-grace-period" in msg
        assert "--warmup-duration" in msg
        # Mentions the request-count / num-sessions alternative path.
        assert "warmup-request-count" in msg or "warmup-num-sessions" in msg


class TestWarmupGracePeriodSuccessPaths:
    """Valid combinations resolve cleanly to a duration-bounded warmup phase."""

    def test_warmup_duration_with_grace_period_resolves(self):
        loadgen = CLIConfig(
            warmup_duration=10.0,
            warmup_grace_period=5.0,
        )
        warmup = build_warmup(_make_user(loadgen))
        assert warmup is not None
        assert warmup["duration"] == 10.0
        assert warmup["grace_period"] == 5.0

    def test_warmup_duration_without_grace_period_resolves(self):
        loadgen = CLIConfig(warmup_duration=10.0)
        warmup = build_warmup(_make_user(loadgen))
        assert warmup is not None
        assert warmup["duration"] == 10.0
        assert "grace_period" not in warmup

    def test_warmup_request_count_without_grace_period_resolves(self):
        """--warmup-request-count alone is valid; only mixing it with
        --warmup-grace-period is the failure mode."""
        loadgen = CLIConfig(warmup_request_count=10)
        warmup = build_warmup(_make_user(loadgen))
        assert warmup is not None
        assert warmup["requests"] == 10
        assert "grace_period" not in warmup

    def test_no_warmup_trigger_returns_none(self):
        """With no warmup_* trigger field set, build_warmup returns None even
        if loadgen.warmup_grace_period happened to be defaulted (None)."""
        loadgen = CLIConfig(request_count=10)
        assert build_warmup(_make_user(loadgen)) is None

    def test_warmup_grace_period_alone_without_any_trigger_raises(self):
        """Ports v1 ``validate_warmup_grace_period``: passing only
        ``--warmup-grace-period`` (no count/sessions/duration trigger) used
        to be a silent no-op (build_warmup returned None and dropped the
        flag). Now it errors so the user discovers the missing trigger.
        """
        loadgen = CLIConfig(warmup_grace_period=5.0)
        with pytest.raises(ValueError, match="--warmup-grace-period.*without any"):
            build_warmup(_make_user(loadgen))
