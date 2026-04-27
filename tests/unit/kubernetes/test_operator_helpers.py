# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests pinning AIPerfJobConfig.to_flat_spec to the v3 CRD shape."""

from __future__ import annotations

from tests.kubernetes.helpers.operator import AIPerfJobConfig


def test_to_flat_spec_uses_profiling_shorthand_not_phases_dict() -> None:
    """phases:{profiling:...} dict shape is invalid against the strict CRD."""
    cfg = AIPerfJobConfig(concurrency=4, request_count=64)
    spec = cfg.to_flat_spec()

    assert "phases" not in spec, (
        "to_flat_spec must NOT emit 'phases:' (mutually exclusive with shorthand)."
    )
    assert spec["profiling"]["type"] == "concurrency"
    assert spec["profiling"]["concurrency"] == 4
    assert spec["profiling"]["requests"] == 64


def test_to_flat_spec_includes_warmup_when_warmup_count_set() -> None:
    cfg = AIPerfJobConfig(concurrency=4, warmup_request_count=5)
    spec = cfg.to_flat_spec()

    assert "warmup" in spec
    assert spec["warmup"]["requests"] == 5
    assert spec["warmup"]["exclude_from_results"] is True


def test_to_flat_spec_omits_warmup_when_warmup_count_zero() -> None:
    cfg = AIPerfJobConfig(concurrency=4, warmup_request_count=0)
    spec = cfg.to_flat_spec()

    assert "warmup" not in spec
