# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.enums module.

Focuses on:
- Case-insensitive enum construction and equality
- JobSetStatus.to_pod_phase mapping
- JobSetStatus.from_str parsing with valid and invalid inputs
- PodPhase.is_retrievable property
- Enum string representation
"""

import pytest
from pytest import param

from aiperf.kubernetes.enums import (
    ImagePullPolicy,
    JobSetStatus,
    PodPhase,
    RestartPolicy,
)

# ============================================================
# RestartPolicy
# ============================================================


class TestRestartPolicy:
    """Verify RestartPolicy members and case-insensitive lookup."""

    def test_members_count(self) -> None:
        assert len(RestartPolicy) == 3

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("Always", RestartPolicy.ALWAYS),
            ("always", RestartPolicy.ALWAYS),
            ("ALWAYS", RestartPolicy.ALWAYS),
            ("OnFailure", RestartPolicy.ON_FAILURE),
            ("onfailure", RestartPolicy.ON_FAILURE),
            ("ONFAILURE", RestartPolicy.ON_FAILURE),
            ("Never", RestartPolicy.NEVER),
            ("never", RestartPolicy.NEVER),
        ],
    )  # fmt: skip
    def test_case_insensitive_lookup(
        self, input_val: str, expected: RestartPolicy
    ) -> None:
        assert RestartPolicy(input_val) == expected

    def test_str_returns_value(self) -> None:
        assert str(RestartPolicy.ON_FAILURE) == "OnFailure"

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            RestartPolicy("Invalid")


# ============================================================
# ImagePullPolicy
# ============================================================


class TestImagePullPolicy:
    """Verify ImagePullPolicy members and case-insensitive lookup."""

    def test_members_count(self) -> None:
        assert len(ImagePullPolicy) == 3

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("Always", ImagePullPolicy.ALWAYS),
            ("always", ImagePullPolicy.ALWAYS),
            ("Never", ImagePullPolicy.NEVER),
            ("IfNotPresent", ImagePullPolicy.IF_NOT_PRESENT),
            ("ifnotpresent", ImagePullPolicy.IF_NOT_PRESENT),
            param("IFNOTPRESENT", ImagePullPolicy.IF_NOT_PRESENT, id="upper-no-underscore"),
        ],
    )  # fmt: skip
    def test_case_insensitive_lookup(
        self, input_val: str, expected: ImagePullPolicy
    ) -> None:
        assert ImagePullPolicy(input_val) == expected

    def test_str_returns_value(self) -> None:
        assert str(ImagePullPolicy.IF_NOT_PRESENT) == "IfNotPresent"

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ImagePullPolicy("PullAlways")


# ============================================================
# JobSetStatus — to_pod_phase
# ============================================================


class TestJobSetStatusToPodPhase:
    """Verify JobSetStatus.to_pod_phase maps every member correctly."""

    @pytest.mark.parametrize(
        "status,expected_phase",
        [
            (JobSetStatus.RUNNING, PodPhase.RUNNING),
            (JobSetStatus.COMPLETED, PodPhase.SUCCEEDED),
            (JobSetStatus.FAILED, PodPhase.FAILED),
            (JobSetStatus.UNKNOWN, PodPhase.UNKNOWN),
        ],
    )  # fmt: skip
    def test_to_pod_phase_mapping(
        self, status: JobSetStatus, expected_phase: PodPhase
    ) -> None:
        assert status.to_pod_phase() == expected_phase

    def test_all_members_have_mapping(self) -> None:
        """Every JobSetStatus member must have a to_pod_phase mapping."""
        for member in JobSetStatus:
            result = member.to_pod_phase()
            assert isinstance(result, PodPhase)


# ============================================================
# JobSetStatus — from_str
# ============================================================


class TestJobSetStatusFromStr:
    """Verify JobSetStatus.from_str parsing."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("Running", JobSetStatus.RUNNING),
            ("running", JobSetStatus.RUNNING),
            ("Completed", JobSetStatus.COMPLETED),
            ("completed", JobSetStatus.COMPLETED),
            ("Failed", JobSetStatus.FAILED),
            ("Unknown", JobSetStatus.UNKNOWN),
        ],
    )  # fmt: skip
    def test_from_str_valid_values(
        self, input_val: str, expected: JobSetStatus
    ) -> None:
        assert JobSetStatus.from_str(input_val) == expected

    @pytest.mark.parametrize(
        "input_val",
        [
            param("Pending", id="not-a-jobset-status"),
            param("", id="empty-string"),
            param("Succeeded", id="pod-phase-not-jobset"),
            param("bogus", id="garbage"),
        ],
    )  # fmt: skip
    def test_from_str_unrecognized_returns_none(self, input_val: str) -> None:
        assert JobSetStatus.from_str(input_val) is None


# ============================================================
# PodPhase — is_retrievable
# ============================================================


class TestPodPhaseIsRetrievable:
    """Verify PodPhase.is_retrievable property."""

    @pytest.mark.parametrize(
        "phase,expected",
        [
            (PodPhase.RUNNING, True),
            (PodPhase.SUCCEEDED, True),
            (PodPhase.PENDING, False),
            (PodPhase.FAILED, False),
            (PodPhase.UNKNOWN, False),
        ],
    )  # fmt: skip
    def test_is_retrievable(self, phase: PodPhase, expected: bool) -> None:
        assert phase.is_retrievable is expected


# ============================================================
# PodPhase — general
# ============================================================


class TestPodPhase:
    """Verify PodPhase members and case-insensitive lookup."""

    def test_members_count(self) -> None:
        assert len(PodPhase) == 5

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("Pending", PodPhase.PENDING),
            ("pending", PodPhase.PENDING),
            ("RUNNING", PodPhase.RUNNING),
            ("succeeded", PodPhase.SUCCEEDED),
            ("Failed", PodPhase.FAILED),
            ("unknown", PodPhase.UNKNOWN),
        ],
    )  # fmt: skip
    def test_case_insensitive_lookup(self, input_val: str, expected: PodPhase) -> None:
        assert PodPhase(input_val) == expected

    def test_str_returns_value(self) -> None:
        assert str(PodPhase.SUCCEEDED) == "Succeeded"

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            PodPhase("Terminating")


# ============================================================
# Cross-enum equality (CaseInsensitiveStrEnum behavior)
# ============================================================


class TestCrossEnumStringEquality:
    """Verify string equality inherited from CaseInsensitiveStrEnum."""

    def test_enum_equals_its_value_string(self) -> None:
        assert JobSetStatus.RUNNING == "Running"

    def test_enum_equals_case_variant_string(self) -> None:
        assert JobSetStatus.RUNNING == "running"

    def test_different_enum_same_value_are_equal(self) -> None:
        """RestartPolicy.ALWAYS and ImagePullPolicy.ALWAYS share value 'Always'."""
        assert RestartPolicy.ALWAYS == ImagePullPolicy.ALWAYS

    def test_enum_not_equal_to_unrelated_string(self) -> None:
        assert PodPhase.RUNNING != "Stopped"
