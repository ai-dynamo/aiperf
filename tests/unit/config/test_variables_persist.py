# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Variables block must persist on the resolved config so run-time renderers can use it."""

from aiperf.config import AIPerfConfig
from aiperf.config.loader import build_benchmark_plan, load_config_from_string
from aiperf.config.loader.jinja import expand_config_dict

_BASE_YAML = """
benchmark:
  models:
    - test/model
  endpoint:
    type: chat
    urls: ["http://localhost:8000"]
  datasets:
    - name: default
      type: synthetic
      entries: 100
      prompts:
        isl: 128
        osl: 64
  phases:
    - name: profiling
      type: concurrency
      requests: 10
      concurrency: 1
"""


_BASE_DICT: dict = {
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
        {"name": "profiling", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
}


def test_variables_block_persists_on_resolved_config():
    yaml_str = (
        """
variables:
  isl: 1024
  osl: 512
"""
        + _BASE_YAML
    )
    config = load_config_from_string(yaml_str)
    assert config.variables == {"isl": 1024, "osl": 512}


def test_variables_default_empty_when_not_declared():
    config = load_config_from_string(_BASE_YAML)
    assert config.variables == {}


def test_variables_block_persists_through_expand_config_dict():
    """K8s/CRD ingestion path: expand_config_dict must keep variables intact.

    Mirrors test_variables_block_persists_on_resolved_config but exercises the
    operator-side dict pipeline used in spec_converter.py rather than the YAML
    string pipeline used by the CLI.
    """
    data = {"variables": {"isl": 1024, "osl": 512}, "benchmark": _BASE_DICT}
    expanded = expand_config_dict(data)

    assert "variables" in expanded
    assert expanded["variables"] == {"isl": 1024, "osl": 512}

    config = AIPerfConfig.model_validate(expanded)
    assert config.variables == {"isl": 1024, "osl": 512}


def test_variables_block_persists_through_sweep_variations():
    """Sweep path: build_benchmark_plan must keep variables on each variation.

    Variables live on the envelope (AIPerfConfig / BenchmarkPlan) in schema-2.0,
    not per-variation BenchmarkConfig. This test locks in that the envelope
    preserves them across sweep expansion.
    """
    config = AIPerfConfig(
        variables={"isl": 1024, "osl": 512},
        sweep={
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [8, 16, 32]},
        },
        benchmark=_BASE_DICT,
    )
    plan = build_benchmark_plan(config)

    assert len(plan.configs) == 3
    assert plan.variables == {"isl": 1024, "osl": 512}


def test_swept_variable_propagates_into_jinja_body_fields():
    """Regression: sweeping `variables.X` must re-render `{{ X }}` in body fields.

    Pre-fix bug: load_config rendered Jinja eagerly, so by the time
    build_benchmark_plan saw the dict the body was already a concrete int
    and the per-variation re-render was a no-op. Variations 0/1/2 all got
    the BASE variable's value.

    Fix: load_config_from_string now stashes the post-env-var, pre-Jinja
    envelope dict on AIPerfConfig._raw_envelope, and build_benchmark_plan
    feeds THAT into expand_sweep so the templates are still live when each
    variation re-renders against its merged variables block.
    """
    yaml_str = """
variables:
  load: 100
sweep:
  type: grid
  parameters:
    variables.load: [10, 50, 100]
benchmark:
  models: [llama]
  endpoint:
    type: chat
    urls: ["http://x:8000/v1/chat/completions"]
  datasets:
    - name: main
      type: synthetic
  phases:
    - name: profiling
      type: concurrency
      requests: "{{ load * 5 }}"
      concurrency: "{{ load }}"
"""
    config = load_config_from_string(yaml_str)
    plan = build_benchmark_plan(config)

    assert len(plan.configs) == 3
    # Each variation's body must reflect its own variables.load value.
    assert [c.phases[0].concurrency for c in plan.configs] == [10, 50, 100]
    assert [c.phases[0].requests for c in plan.configs] == [50, 250, 500]


def test_static_jinja_body_unaffected_by_unrelated_sweep():
    """Sanity: a static `{{ var }}` body field still resolves correctly when
    the swept dimension is something other than `variables.*`.

    Pinned alongside the swept-variable regression so a future refactor that
    breaks ONE direction will be caught even if the OTHER stays green.
    """
    yaml_str = """
variables:
  base_load: 200
sweep:
  type: grid
  parameters:
    phases.profiling.concurrency: [1, 2]
benchmark:
  models: [llama]
  endpoint:
    type: chat
    urls: ["http://x:8000/v1/chat/completions"]
  datasets:
    - name: main
      type: synthetic
  phases:
    - name: profiling
      type: concurrency
      requests: "{{ base_load * 5 }}"
      concurrency: 1
"""
    config = load_config_from_string(yaml_str)
    plan = build_benchmark_plan(config)

    assert len(plan.configs) == 2
    # base_load is not swept -> requests=1000 in every variation.
    assert all(c.phases[0].requests == 1000 for c in plan.configs)
    # The bare-path sweep on concurrency still wins over the YAML default.
    assert [c.phases[0].concurrency for c in plan.configs] == [1, 2]
