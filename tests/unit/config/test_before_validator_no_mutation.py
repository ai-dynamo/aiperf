# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests: ``mode="before"`` validators must not mutate their input.

A pydantic ``mode="before"`` validator receives the CALLER's actual dict, so
normalizing in place leaks renamed/hoisted keys back into caller state (kopf
resource bodies, fixtures, any dict validated then reused). A shallow copy is
not enough on its own: any nested dict the validator writes into (e.g. the
``prompts`` sub-dict of a synthetic dataset) must be copied too, or the shallow
copy still shares — and mutates — the caller's nested object.

The invariant under test: ``load(d)`` leaves ``d`` byte-for-byte unchanged, and
a second ``model_validate`` of the same caller dict (after the caller edits a
visible top-level field) reflects that edit rather than stale hoisted values.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pytest import param

from aiperf.config.config import BenchmarkConfig
from aiperf.config.dataset.config import SyntheticDataset
from aiperf.config.endpoint import EndpointConfig
from aiperf.config.gpu_telemetry import GpuTelemetryConfig
from aiperf.config.loader.normalizers import parse_datasets_input
from aiperf.config.server_metrics import ServerMetricsConfig
from aiperf.config.types import SequenceDistributionEntry


@pytest.mark.parametrize(
    ("model", "data"),
    [
        param(
            SyntheticDataset,
            {"name": "main", "type": "synthetic", "isl": 100, "prompts": {"osl": 50}},
            id="synthetic_dataset_isl_osl_hoist_nested_prompts",
        ),
        param(
            SyntheticDataset,
            {"name": "main", "type": "synthetic", "isl": 100, "osl": 50},
            id="synthetic_dataset_isl_osl_hoist_no_prompts",
        ),
        param(
            EndpointConfig,
            {"url": "http://host:8000"},
            id="endpoint_url_to_urls",
        ),
        param(
            SequenceDistributionEntry,
            {
                "isl": {"mean": 512, "stddev": 30},
                "osl": 64,
                "osl_stddev": 25,
                "probability": 100,
            },
            id="sequence_distribution_stddev_shorthand_nested_isl",
        ),
        param(
            SequenceDistributionEntry,
            {"isl": 128, "isl_stddev": 50, "osl": 64, "osl_stddev": 25, "probability": 100},
            id="sequence_distribution_stddev_shorthand_scalars",
        ),
        param(
            ServerMetricsConfig,
            {"url": "http://host:9090", "discovery": {"namespace": "ns"}},
            id="server_metrics_url_to_urls_nested_discovery",
        ),
        param(
            GpuTelemetryConfig,
            {"url": "http://host:9400"},
            id="gpu_telemetry_url_to_urls",
        ),
    ],
)  # fmt: skip
def test_before_validator_does_not_mutate_input(
    model: type, data: dict[str, Any]
) -> None:
    """model_validate must leave the caller's input dict (and its nested dicts) untouched."""
    snapshot = copy.deepcopy(data)
    model.model_validate(data)
    assert data == snapshot, (
        f"{model.__name__}.model_validate mutated its input: "
        f"before={snapshot!r} after={data!r}"
    )


def test_parse_datasets_input_does_not_mutate_list_or_nested_prompts() -> None:
    """The list-level dataset hoist must not mutate the caller's list or nested prompts.

    Covers ``_normalize_single_dataset_listed`` -> ``_hoist_synthetic_prompt_fields``,
    the shallow-copy site that previously left the shared ``prompts`` sub-dict
    exposed to in-place ``setdefault``.
    """
    data = [{"name": "main", "type": "synthetic", "isl": 100, "prompts": {"osl": 50}}]
    snapshot = copy.deepcopy(data)
    parse_datasets_input(data)
    assert data == snapshot


def test_benchmark_config_does_not_mutate_input_datasets() -> None:
    """End-to-end: BenchmarkConfig.model_validate must not mutate a nested datasets dict."""
    data = {
        "model": "llama",
        "endpoint": {"url": "http://host:8000"},
        "profiling": {"type": "concurrency", "concurrency": 1, "requests": 10},
        "datasets": [
            {"name": "main", "type": "synthetic", "isl": 100, "prompts": {"osl": 50}}
        ],
    }
    snapshot = copy.deepcopy(data)
    BenchmarkConfig.model_validate(data)
    assert data == snapshot


def test_synthetic_dataset_reload_reflects_top_level_edit() -> None:
    """Re-loading the same caller dict after editing a top-level field must reflect it.

    Before the nested-copy fix, the first model_validate hoisted ``isl`` into the
    shared ``prompts`` sub-dict, so a later edit to top-level ``isl`` was masked by
    the already-hoisted stale value (``setdefault`` no-ops on the populated key).
    """
    data = {"name": "main", "type": "synthetic", "isl": 100, "prompts": {}}

    first = SyntheticDataset.model_validate(data)
    assert first.prompts is not None
    assert first.prompts.isl.value == 100.0

    data["isl"] = 200
    second = SyntheticDataset.model_validate(data)
    assert second.prompts is not None
    assert second.prompts.isl.value == 200.0
