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
from aiperf.config.config import BenchmarkConfig
from aiperf.config.distributions import (
    Distribution,
    EmpiricalDistribution,
    FixedDistribution,
    LogNormalDistribution,
    MultimodalDistribution,
    NormalDistribution,
)
from aiperf.config.endpoint import EndpointConfig
from aiperf.config.gpu_telemetry import GpuTelemetryConfig
from aiperf.config.server_metrics import ServerMetricsConfig

PRESERVE = "x-kubernetes-preserve-unknown-fields"


def test_aiperf_config_models_field_marks_preserve_unknown_fields():
    """BenchmarkConfig.models accepts str | list[str] | ModelsAdvanced — must mark preserve-unknown.

    The marker lives on the field-level property schema (which sits beside the
    ``$ref`` to ModelsAdvanced); it is the field-level extras dict that the CRD
    generator picks up when it walks BenchmarkConfig.
    """
    schema = BenchmarkConfig.model_json_schema()
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


def test_aiperf_config_schema_exposes_top_level_shortcuts():
    """model/dataset/warmup/profiling appear as optional schema siblings with preserve-unknown.

    These shortcuts live on BenchmarkConfig (the swept body), not on
    the AIPerfConfig envelope.
    """
    schema = BenchmarkConfig.model_json_schema()
    props = schema["properties"]
    for key in ("model", "dataset", "warmup", "profiling"):
        assert key in props, f"BenchmarkConfig schema missing shortcut sibling {key!r}"
        assert props[key].get(PRESERVE) is True, (
            f"shortcut {key!r} must mark {PRESERVE}=true; got: {props[key]!r}"
        )


def test_aiperf_config_runtime_still_validates_with_string_model_shortcut():
    """The before-validator hoist is unchanged — passing model: 'foo' still works after the field is exposed."""
    cfg = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "model": "test-model",
                "endpoint": {"type": "chat", "url": "http://x:8000"},
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 1,
                    }
                ],
                "datasets": [{"name": "profiling", "type": "synthetic"}],
            }
        }
    )
    assert cfg.benchmark.models.items[0].name == "test-model"


def test_aiperf_config_dump_excludes_shortcut_siblings():
    """Adding the shortcut as a real field must not pollute model_dump output."""
    cfg = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "model": "test-model",
                "endpoint": {"type": "chat", "url": "http://x:8000"},
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 1,
                    }
                ],
                "datasets": [{"name": "profiling", "type": "synthetic"}],
            }
        }
    )
    dumped = cfg.model_dump(exclude_none=True)
    for shortcut in ("model", "dataset", "warmup", "profiling"):
        assert shortcut not in dumped.get("benchmark", {}), (
            f"{shortcut!r} leaked into canonical model_dump; the field must be exclude=True"
        )


def test_synthetic_dataset_schema_exposes_isl_osl_shortcuts():
    """SyntheticDataset exposes isl/osl as optional schema siblings with preserve-unknown."""
    from aiperf.config.dataset import SyntheticDataset

    schema = SyntheticDataset.model_json_schema()
    props = schema["properties"]
    for key in ("isl", "osl"):
        assert key in props, f"SyntheticDataset schema missing shortcut sibling {key!r}"
        assert props[key].get(PRESERVE) is True, (
            f"shortcut {key!r} must mark {PRESERVE}=true; got: {props[key]!r}"
        )


def test_synthetic_dataset_runtime_still_validates_with_top_level_isl():
    """Before-validator hoist (isl -> prompts.isl) is unchanged after the field is exposed."""
    from aiperf.config.dataset import SyntheticDataset

    ds = SyntheticDataset.model_validate(
        {
            "name": "test",
            "type": "synthetic",
            "isl": 512,
            "osl": 128,
        }
    )
    # The shortcut should have been hoisted; verify by checking the canonical location.
    assert ds.prompts is not None
    isl_val = ds.prompts.isl
    osl_val = ds.prompts.osl
    # SamplingDistribution may wrap a scalar; check both raw and .value
    assert isl_val == 512 or getattr(isl_val, "value", None) == 512, (
        f"prompts.isl expected 512, got {isl_val!r}"
    )
    assert osl_val == 128 or getattr(osl_val, "value", None) == 128, (
        f"prompts.osl expected 128, got {osl_val!r}"
    )


def test_synthetic_dataset_dump_excludes_isl_osl_shortcuts():
    """isl/osl shortcuts must not pollute SyntheticDataset.model_dump output."""
    from aiperf.config.dataset import SyntheticDataset

    ds = SyntheticDataset.model_validate(
        {
            "name": "test",
            "type": "synthetic",
            "isl": 512,
            "osl": 128,
        }
    )
    dumped = ds.model_dump(exclude_none=True)
    for shortcut in ("isl", "osl"):
        assert shortcut not in dumped, (
            f"{shortcut!r} leaked into canonical SyntheticDataset.model_dump; "
            f"the field must be exclude=True"
        )
