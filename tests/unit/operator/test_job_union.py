# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the unified-jobs-source helpers."""

from __future__ import annotations

import pytest

from aiperf.kubernetes.models import AIPerfJobInfo


def test_aiperfjobinfo_source_defaults_to_live():
    info = AIPerfJobInfo(
        name="j1", namespace="ns", phase="Running", job_id="j1",
    )
    assert info.source == "live"


def test_aiperfjobinfo_source_accepts_archived_and_both():
    for s in ("archived", "both"):
        info = AIPerfJobInfo(
            name="j1", namespace="ns", phase="Succeeded", job_id="j1", source=s,
        )
        assert info.source == s


def test_aiperfjobinfo_source_rejects_unknown():
    with pytest.raises(ValueError):
        AIPerfJobInfo(
            name="j1", namespace="ns", phase="Running", job_id="j1", source="bogus",
        )
