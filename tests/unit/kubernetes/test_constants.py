# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanity tests for aiperf.kubernetes.constants.

These are frozen dataclass constants with no logic.  Tests focus on:
- All values are non-empty strings (a typo producing "" would break selectors/labels)
- No duplicate values within a dataclass (copy-paste errors)
- Instances are truly frozen (consumers rely on immutability)
- SELECTOR label is consistent with APP_KEY/APP_VALUE
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
from pytest import param

from aiperf.kubernetes.constants import (
    AIPerfLabels,
    Annotations,
    Containers,
    JobSetLabels,
    KueueLabels,
    ProgressAnnotations,
)

_ALL_CONSTANT_CLASSES = [
    param(JobSetLabels, id="JobSetLabels"),
    param(AIPerfLabels, id="AIPerfLabels"),
    param(Annotations, id="Annotations"),
    param(ProgressAnnotations, id="ProgressAnnotations"),
    param(Containers, id="Containers"),
    param(KueueLabels, id="KueueLabels"),
]  # fmt: skip


class TestConstantIntegrity:
    """Verify that constant dataclasses have non-empty, unique values."""

    @pytest.mark.parametrize("cls", _ALL_CONSTANT_CLASSES)  # fmt: skip
    def test_all_values_are_nonempty_strings(self, cls: type) -> None:
        instance = cls()
        for f in fields(instance):
            val = getattr(instance, f.name)
            assert isinstance(val, str), f"{cls.__name__}.{f.name} is not a str"
            assert val != "", f"{cls.__name__}.{f.name} is empty"

    @pytest.mark.parametrize("cls", _ALL_CONSTANT_CLASSES)  # fmt: skip
    def test_no_duplicate_values(self, cls: type) -> None:
        instance = cls()
        values = [getattr(instance, f.name) for f in fields(instance)]
        assert len(values) == len(set(values)), (
            f"{cls.__name__} has duplicate values: {values}"
        )

    @pytest.mark.parametrize("cls", _ALL_CONSTANT_CLASSES)  # fmt: skip
    def test_frozen_immutability(self, cls: type) -> None:
        instance = cls()
        first_field = fields(instance)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(instance, first_field, "tampered")


class TestLabelsConsistency:
    """Verify internal consistency of the AIPerfLabels dataclass."""

    def test_selector_matches_app_key_and_value(self) -> None:
        labels = AIPerfLabels()
        expected = f"{labels.APP_KEY}={labels.APP_VALUE}"
        assert expected == labels.SELECTOR


class TestNamespacePrefixes:
    """Verify that aiperf-owned labels/annotations use the correct domain prefix."""

    @pytest.mark.parametrize(
        "cls,field_name",
        [
            param(AIPerfLabels, "JOB_ID", id="AIPerfLabels.JOB_ID"),
            param(AIPerfLabels, "NAME", id="AIPerfLabels.NAME"),
            param(Annotations, "MODEL", id="Annotations.MODEL"),
            param(Annotations, "ENDPOINT", id="Annotations.ENDPOINT"),
            param(ProgressAnnotations, "PHASE", id="ProgressAnnotations.PHASE"),
            param(ProgressAnnotations, "PERCENT", id="ProgressAnnotations.PERCENT"),
            param(ProgressAnnotations, "REQUESTS", id="ProgressAnnotations.REQUESTS"),
            param(ProgressAnnotations, "STATUS", id="ProgressAnnotations.STATUS"),
        ],
    )  # fmt: skip
    def test_aiperf_fields_use_nvidia_domain_prefix(
        self, cls: type, field_name: str
    ) -> None:
        value = getattr(cls(), field_name)
        assert value.startswith("aiperf.nvidia.com/"), (
            f"{cls.__name__}.{field_name} = {value!r} missing domain prefix"
        )

    @pytest.mark.parametrize(
        "field_name",
        [
            param("POD_INDEX", id="POD_INDEX"),
            param("JOBSET_NAME", id="JOBSET_NAME"),
            param("REPLICATED_JOB_NAME", id="REPLICATED_JOB_NAME"),
        ],
    )  # fmt: skip
    def test_jobset_labels_use_jobset_domain_prefix(self, field_name: str) -> None:
        value = getattr(JobSetLabels(), field_name)
        assert value.startswith("jobset.sigs.k8s.io/"), (
            f"JobSetLabels.{field_name} = {value!r} missing jobset domain prefix"
        )

    @pytest.mark.parametrize(
        "field_name",
        [
            param("QUEUE_NAME", id="QUEUE_NAME"),
            param("PRIORITY_CLASS", id="PRIORITY_CLASS"),
        ],
    )  # fmt: skip
    def test_kueue_labels_use_kueue_domain_prefix(self, field_name: str) -> None:
        value = getattr(KueueLabels(), field_name)
        assert value.startswith("kueue.x-k8s.io/"), (
            f"KueueLabels.{field_name} = {value!r} missing kueue domain prefix"
        )
