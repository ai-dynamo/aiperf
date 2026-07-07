# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanity check: every bundled YAML template validates as an AIPerfConfig.

This locks in the schema-2.0 envelope shape across the entire
`src/aiperf/config/templates/` directory. If a new template is added that
puts body fields at the top level (pre-restructure flat shape) or fails
validation for any other reason, this test fires per-file.
"""

from __future__ import annotations

import os
import pathlib

import pytest
import yaml

from aiperf.config.loader.core import load_config
from aiperf.config.loader.plan import build_benchmark_plan

TEMPLATES_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "config"
    / "templates"
)

# Defaults for env vars referenced by `${VAR:default}` substitutions in some
# templates (env_var_production, jinja2_variables, scenario_workload_profiles,
# sweep_distributions). Templates use sensible defaults already, but we set
# stable values here so the test is hermetic regardless of host env.
_TEMPLATE_ENV_DEFAULTS = {
    "MODEL_NAME": "meta-llama/Llama-3.1-8B-Instruct",
    "INFERENCE_URL": "http://localhost:8000/v1/chat/completions",
    "METRICS_URL": "http://localhost:8000/metrics",
    "TIMEOUT": "600.0",
    "BENCHMARK_SEED": "42",
    "DURATION": "300",
    "TARGET_RATE": "30.0",
    "MAX_CONCURRENCY": "64",
    "NUM_RUNS": "3",
    "COOLDOWN": "30",
    "ARTIFACTS_DIR": "./artifacts/test",
}


@pytest.fixture(autouse=True)
def _set_template_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide stable env-var values for templates that use ${VAR:default}."""
    for key, value in _TEMPLATE_ENV_DEFAULTS.items():
        monkeypatch.setenv(key, os.environ.get(key, value))


@pytest.mark.parametrize(
    "template_path",
    sorted(TEMPLATES_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_bundled_template_validates_as_aiperf_config(
    template_path: pathlib.Path,
) -> None:
    """Every bundled template loads + validates via the AIPerfConfig envelope.

    Failure means the template puts body fields at the top level (legacy
    flat shape) or otherwise fails schema validation. Migrate body fields
    under `benchmark:` and keep envelope keys (sweep, multi_run, variables,
    random_seed) at the top level.
    """
    load_config(template_path)


def test_bundled_templates_directory_is_non_empty() -> None:
    """Guards against an accidental empty glob hiding a regression."""
    assert sorted(TEMPLATES_DIR.glob("*.yaml")), (
        f"No bundled templates found under {TEMPLATES_DIR}; the parametrized "
        "validation test would have silently passed with zero cases."
    )


def _has_sweep_block(template_path: pathlib.Path) -> bool:
    """True when the template declares a top-level `sweep:` block."""
    data = yaml.safe_load(template_path.read_text())
    return isinstance(data, dict) and data.get("sweep") is not None


_SWEEP_TEMPLATES = sorted(
    p for p in TEMPLATES_DIR.glob("*.yaml") if _has_sweep_block(p)
)


@pytest.mark.parametrize(
    "template_path",
    _SWEEP_TEMPLATES,
    ids=lambda p: p.name,
)
def test_bundled_sweep_template_expands_via_build_benchmark_plan(
    template_path: pathlib.Path,
) -> None:
    """Every bundled sweep template must expand, not just load.

    ``load_config`` alone never runs the per-variation ``BenchmarkConfig``
    validation that ``aiperf profile`` / ``aiperf config expand`` do. A broken
    sweep (e.g. singular ``dataset:`` combined with a ``datasets.*`` sweep path,
    as ``sweep_distributions.yaml`` once shipped) passes ``load_config`` yet
    crashes on use. Expanding the plan here catches that class of regression.
    """
    config = load_config(template_path)
    plan = build_benchmark_plan(config)
    assert len(plan.configs) >= 1


def test_sweep_distributions_expands_to_full_grid() -> None:
    """`sweep_distributions.yaml` is a 3 ISL x 3 rate grid == 9 variations."""
    config = load_config(TEMPLATES_DIR / "sweep_distributions.yaml")
    plan = build_benchmark_plan(config)
    assert len(plan.configs) == 9
