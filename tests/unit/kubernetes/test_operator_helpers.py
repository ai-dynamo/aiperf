# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests pinning AIPerfJobConfig.to_flat_spec to the v3 CRD shape."""

from __future__ import annotations

from tests.kubernetes.helpers.operator import AIPerfJobConfig


def test_to_flat_spec_emits_phases_array_with_profiling_entry() -> None:
    cfg = AIPerfJobConfig(concurrency=4, request_count=64)
    spec = cfg.to_flat_spec()

    assert isinstance(spec["phases"], list)
    profiling = next((p for p in spec["phases"] if p["name"] == "profiling"), None)
    assert profiling is not None
    assert profiling["type"] == "concurrency"
    assert profiling["concurrency"] == 4
    assert profiling["requests"] == 64
    # Must NOT use top-level shorthand; the CRD's apiserver validator
    # rejects bodies that omit `phases:`.
    assert "profiling" not in spec
    assert "warmup" not in spec


def test_to_flat_spec_warmup_appears_before_profiling_when_set() -> None:
    cfg = AIPerfJobConfig(concurrency=4, warmup_request_count=5)
    spec = cfg.to_flat_spec()

    names = [p["name"] for p in spec["phases"]]
    assert names == ["warmup", "profiling"]
    warmup = spec["phases"][0]
    assert warmup["requests"] == 5
    assert warmup["exclude_from_results"] is True


def test_to_flat_spec_omits_warmup_when_count_zero() -> None:
    cfg = AIPerfJobConfig(concurrency=4, warmup_request_count=0)
    spec = cfg.to_flat_spec()

    names = [p["name"] for p in spec["phases"]]
    assert names == ["profiling"]


def test_to_flat_spec_omits_artifacts_records_for_summary_only_default() -> None:
    """Per-record records export (jsonl/csv) was previously enabled here,
    but combining records + a sub-second benchmark hits a controller-side
    race where the readiness marker is never written and the operator
    marks the job Failed. Audit-parity uses summary-only; the bare side
    likewise drops --export-level records."""
    cfg = AIPerfJobConfig(concurrency=4, request_count=64)
    spec = cfg.to_flat_spec()

    assert "artifacts" not in spec or "records" not in spec.get("artifacts", {})
