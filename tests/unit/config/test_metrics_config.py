# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.enums import ListMetricAggregationMode
from aiperf.config import AIPerfConfig


def _minimal_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "models": ["test-model"],
        "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
        "datasets": {"main": {"type": "synthetic"}},
        "phases": {"default": {"type": "concurrency", "requests": 1}},
    }
    config.update(overrides)
    return config


def test_metrics_config_defaults_to_exact_aggregation() -> None:
    config = AIPerfConfig.model_validate(_minimal_config())

    assert config.metrics is not None
    assert config.metrics.list_metric_aggregation == ListMetricAggregationMode.EXACT


def test_metrics_config_accepts_tdigest_camel_case_alias() -> None:
    config = AIPerfConfig.model_validate(
        _minimal_config(metrics={"listMetricAggregation": "tdigest"})
    )

    assert config.metrics is not None
    assert config.metrics.list_metric_aggregation == ListMetricAggregationMode.TDIGEST


def test_metrics_config_rejects_invalid_aggregation_mode() -> None:
    with pytest.raises(ValidationError) as exc_info:
        AIPerfConfig.model_validate(
            _minimal_config(metrics={"listMetricAggregation": "invalid"})
        )

    assert "listMetricAggregation" in str(exc_info.value)


def test_metrics_config_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError) as exc_info:
        AIPerfConfig.model_validate(_minimal_config(metrics={"unexpectedField": True}))

    assert "unexpectedField" in str(exc_info.value)
