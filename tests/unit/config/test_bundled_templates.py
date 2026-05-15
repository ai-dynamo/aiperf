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

from aiperf.config.loader.core import load_config

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
