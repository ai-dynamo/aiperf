# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for jobs-router response models, including sweep-linkage fields."""

from __future__ import annotations

from aiperf.operator.routers.jobs_models import ActiveJobSummary


def _minimum_fields() -> dict:
    """Return the minimum required-field kwargs for ActiveJobSummary.

    Update this dict if upstream adds new required fields; the tests below
    do not care about anything except the three new optional sweep fields.
    """
    return {
        "namespace": "bench",
        "name": "ch-0-0",
        "phase": "Succeeded",
        "job_id": "ch-0-0",
        "source": "live",
    }


def test_active_job_summary_sweep_fields_default_none() -> None:
    s = ActiveJobSummary(**_minimum_fields())
    assert s.sweep_name is None
    assert s.variation_index is None
    assert s.variation_label is None


def test_active_job_summary_sweep_fields_round_trip_via_alias() -> None:
    s = ActiveJobSummary(
        **_minimum_fields(),
        sweep_name="saturation-sweep",
        variation_index=7,
        variation_label="concurrency-128-rate-50",
    )
    payload = s.model_dump(by_alias=True)
    assert payload["sweepName"] == "saturation-sweep"
    assert payload["variationIndex"] == 7
    assert payload["variationLabel"] == "concurrency-128-rate-50"
