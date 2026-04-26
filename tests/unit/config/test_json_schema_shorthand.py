# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify shorthand-accepting fields emit x-kubernetes-preserve-unknown-fields in JSON schema.

Kubernetes structural schemas can't express mixed-type unions (string | list[str] |
object). Each shorthand-accepting boundary must emit
``x-kubernetes-preserve-unknown-fields: true`` so the apiserver lets the subtree
through to ``AIPerfConfig.model_validate`` for runtime normalization.
"""

from __future__ import annotations

import pytest

from aiperf.config import AIPerfConfig
from aiperf.config.artifacts import GpuTelemetryConfig, ServerMetricsConfig
from aiperf.config.distributions import (
    Distribution,
    EmpiricalDistribution,
    FixedDistribution,
    LogNormalDistribution,
    MultimodalDistribution,
    NormalDistribution,
)
from aiperf.config.endpoint import EndpointConfig

PRESERVE = "x-kubernetes-preserve-unknown-fields"


def test_aiperf_config_models_field_marks_preserve_unknown_fields():
    """AIPerfConfig.models accepts str | list[str] | ModelsAdvanced — must mark preserve-unknown.

    The marker lives on the field-level property schema (which sits beside the
    ``$ref`` to ModelsAdvanced); it is the field-level extras dict that the CRD
    generator picks up when it walks AIPerfConfig.
    """
    schema = AIPerfConfig.model_json_schema()
    models_prop = schema["properties"]["models"]
    assert models_prop.get(PRESERVE) is True, (
        f"models field must mark {PRESERVE}=true (it accepts str/list/object shorthand); "
        f"got: {models_prop!r}"
    )


def test_endpoint_config_urls_marks_preserve_unknown_fields():
    """EndpointConfig.urls must mark preserve-unknown to allow url-singular shorthand."""
    schema = EndpointConfig.model_json_schema()
    urls_schema = schema["properties"].get("urls", {})
    assert urls_schema.get(PRESERVE) is True, (
        f"EndpointConfig.urls must mark {PRESERVE}=true (accepts str | list[str] via "
        f"url->urls before-validator); got: {urls_schema!r}"
    )


@pytest.mark.parametrize("cls", [ServerMetricsConfig, GpuTelemetryConfig])
def test_telemetry_config_marks_preserve_unknown_fields_class_level(cls):
    """ServerMetricsConfig/GpuTelemetryConfig accept string URL shorthand at the class level."""
    schema = cls.model_json_schema()
    assert schema.get(PRESERVE) is True, (
        f"{cls.__name__} must mark {PRESERVE}=true at class level "
        f"(accepts string URL or url-singular shorthand); got top-level keys: "
        f"{list(schema.keys())}"
    )


def test_distribution_subclasses_mark_preserve_unknown_fields():
    """Every concrete Distribution subclass must mark preserve-unknown.

    FixedDistribution coerces int|float scalars in its before-validator; the rest
    inherit the marker via the Distribution base class. Either the base class
    itself emits the marker (Pydantic propagates json_schema_extra to subclasses)
    or each concrete subclass does.
    """
    base_schema = Distribution.model_json_schema()
    if base_schema.get(PRESERVE) is True:
        # Base-class marker propagates — sufficient.
        return

    # Otherwise every concrete subclass must mark it explicitly.
    for sub in (
        FixedDistribution,
        NormalDistribution,
        LogNormalDistribution,
        MultimodalDistribution,
        EmpiricalDistribution,
    ):
        sub_schema = sub.model_json_schema()
        assert sub_schema.get(PRESERVE) is True, (
            f"{sub.__name__} must mark {PRESERVE}=true (accepts scalar shorthand "
            f"via FixedDistribution coerce_scalar / discriminated union); "
            f"got top-level keys: {list(sub_schema.keys())}"
        )
