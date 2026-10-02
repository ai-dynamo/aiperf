# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A rejected AIPerfSweep spec must say so on the CR, not only in the logs.

kopf's ``PermanentError`` stops the retry loop but writes nothing to status.
The sweep create handler used to raise it bare, so an invalid AIPerfSweep sat
with a completely empty status object -- blank phase in ``kubectl get``, no
conditions, no error -- indistinguishable from one the operator never saw.
AIPerfJob has always surfaced this; these tests pin the parity.
"""

from __future__ import annotations

import kopf
import pytest

from aiperf.operator.handlers.sweep.create import _record_spec_invalid


@pytest.fixture
def patch() -> kopf.Patch:
    return kopf.Patch()


class TestSweepSpecInvalidStatus:
    def test_sets_failed_phase_and_error(self, patch: kopf.Patch) -> None:
        _record_spec_invalid(patch, ValueError("sweep.grid.parameters missing"))
        assert patch.status["phase"] == "Failed"
        assert "sweep.grid.parameters missing" in patch.status["error"]
        assert patch.status["completionTime"]

    def test_sets_config_valid_false_with_the_reason(self, patch: kopf.Patch) -> None:
        _record_spec_invalid(patch, ValueError("bad distribution"))
        by_type = {c["type"]: c for c in patch.status["conditions"]}

        assert by_type["ConfigValid"]["status"] == "False"
        assert by_type["ConfigValid"]["reason"] == "SpecInvalid"
        assert "bad distribution" in by_type["ConfigValid"]["message"]

    def test_sets_failed_condition_true(self, patch: kopf.Patch) -> None:
        """`kubectl wait --for=condition=Failed` should resolve on a bad spec."""
        _record_spec_invalid(patch, ValueError("nope"))
        by_type = {c["type"]: c for c in patch.status["conditions"]}
        assert by_type["Failed"]["status"] == "True"

    def test_every_condition_carries_a_transition_time(self, patch: kopf.Patch) -> None:
        _record_spec_invalid(patch, ValueError("nope"))
        assert all(c.get("lastTransitionTime") for c in patch.status["conditions"])

    def test_truncates_a_pathological_validation_error(self, patch: kopf.Patch) -> None:
        """Pydantic can emit enormous errors; status has a hard size budget."""
        _record_spec_invalid(patch, ValueError("x" * 100_000))
        for condition in patch.status["conditions"]:
            assert len(condition["message"]) <= 32768
