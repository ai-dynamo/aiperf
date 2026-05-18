# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compound variables: ``variables:`` entries may reference each other.

Resolution is dependency-ordered (YAML position is irrelevant), cycles raise
ConfigurationError, and rendered values are type-coerced like every other
load-time jinja2 expression.
"""

from __future__ import annotations

import pytest

from aiperf.config.loader.errors import ConfigurationError
from aiperf.config.loader.jinja import build_template_context, render_jinja2_templates


def _expand(data: dict) -> dict:
    """Run the same two-step pipeline as expand_config_dict (no env subst)."""
    context = build_template_context(data)
    return render_jinja2_templates(data, context)


def _phase_by_name(phases: list[dict], name: str) -> dict:
    """Find the phase entry in a list-shaped phases by ``name``."""
    return next(p for p in phases if p["name"] == name)


def test_variable_references_another_variable_resolved_value() -> None:
    """Compound variable visible in templated phase fields (list shape)."""
    data = {
        "variables": {
            "concurrency_per_gpu": 30,
            "deployment_gpu_count": 4,
            "total_concurrency": "{{ concurrency_per_gpu * deployment_gpu_count }}",
        },
        "phases": [{"name": "profiling", "concurrency": "{{ total_concurrency }}"}],
    }
    result = _expand(data)
    assert result["variables"]["total_concurrency"] == 120
    assert _phase_by_name(result["phases"], "profiling")["concurrency"] == 120


def test_variable_resolution_is_order_independent() -> None:
    """Forward refs in YAML order must still resolve via dep graph."""
    data = {
        "variables": {
            "total": "{{ a * b }}",  # defined BEFORE its dependencies in YAML
            "a": 30,
            "b": 4,
        },
    }
    result = _expand(data)
    assert result["variables"]["total"] == 120


def test_variable_chain_three_deep() -> None:
    data = {
        "variables": {
            "a": 2,
            "b": "{{ a * 3 }}",
            "c": "{{ b + 4 }}",
        },
    }
    result = _expand(data)
    assert result["variables"] == {"a": 2, "b": 6, "c": 10}


def test_variable_cycle_raises_configuration_error() -> None:
    data = {
        "variables": {
            "a": "{{ b }}",
            "b": "{{ a }}",
        },
    }
    with pytest.raises(ConfigurationError) as exc_info:
        _expand(data)
    msg = str(exc_info.value.message)
    assert "circular" in msg.lower() or "cycle" in msg.lower()
    assert "a" in msg and "b" in msg


def test_variable_self_reference_is_a_cycle() -> None:
    data = {"variables": {"a": "{{ a + 1 }}"}}
    with pytest.raises(ConfigurationError):
        _expand(data)


def test_variable_resolved_values_are_type_coerced() -> None:
    """Variable values rendered from templates use the same int/float/bool coercion."""
    data = {
        "variables": {
            "n": 10,
            "doubled": "{{ n * 2 }}",  # int
            "halved": "{{ n / 4 }}",  # float
            "is_big": "{{ n > 5 }}",  # bool
        },
    }
    result = _expand(data)
    assert result["variables"]["doubled"] == 20
    assert isinstance(result["variables"]["doubled"], int)
    assert result["variables"]["halved"] == 2.5
    assert isinstance(result["variables"]["halved"], float)
    assert result["variables"]["is_big"] is True


def test_variable_can_reference_top_level_config_field() -> None:
    """A variable may reference any flattened path from the rest of the config.

    For list-shaped phases, ``phases.<name>.<field>`` resolves to the entry
    in the list whose ``name`` matches.
    """
    data = {
        "variables": {
            "default_conc": "{{ phases.warmup.concurrency * 4 }}",
        },
        "phases": [{"name": "warmup", "concurrency": 8}],
    }
    result = _expand(data)
    assert result["variables"]["default_conc"] == 32


def test_undefined_variable_in_variables_block_raises() -> None:
    data = {
        "variables": {
            "a": "{{ does_not_exist }}",
        },
    }
    with pytest.raises(ConfigurationError) as exc_info:
        _expand(data)
    assert "does_not_exist" in str(exc_info.value.message)


def test_non_template_variables_pass_through_unchanged() -> None:
    data = {"variables": {"a": 1, "b": "literal", "c": [1, 2, 3]}}
    result = _expand(data)
    assert result["variables"] == {"a": 1, "b": "literal", "c": [1, 2, 3]}


def test_empty_variables_block_does_not_break() -> None:
    data = {"variables": {}, "phases": [{"name": "warmup", "concurrency": 1}]}
    result = _expand(data)
    assert result["variables"] == {}
    assert result["phases"][0]["concurrency"] == 1


def test_resolved_variables_visible_in_run_time_user_files_context() -> None:
    """End-to-end: a compound variable resolved at load-time persists into the
    BenchmarkConfig.variables dict, so artifacts.user_files at run-time sees the
    final value (120), not the raw template."""
    from aiperf.config import AIPerfConfig
    from aiperf.config.loader.jinja import expand_config_dict

    raw = {
        "variables": {
            "concurrency_per_gpu": 30,
            "deployment_gpu_count": 4,
            "total_concurrency": "{{ concurrency_per_gpu * deployment_gpu_count }}",
        },
        "benchmark": {
            "models": ["test/model"],
            "endpoint": {"type": "chat", "urls": ["http://localhost:8000"]},
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 128, "osl": 64},
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        },
    }
    expanded = expand_config_dict(raw, substitute_env=False)
    config = AIPerfConfig.model_validate(expanded)
    assert config.variables["total_concurrency"] == 120
