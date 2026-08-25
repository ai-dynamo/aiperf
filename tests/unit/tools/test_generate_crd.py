# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip and CEL-invariant checks for the auto-generated CRD YAML.

The CRDs under ``deploy/helm/aiperf-operator/templates/`` are generated from
``AIPerfJobSpec`` / ``AIPerfSweepSpec`` by ``tools/generate_crd.py`` and must
never be hand-edited. These tests fail loudly when the checked-in YAML drifts
from the Pydantic models, and when the kind-specific ``spec.sweep`` CEL rules
that distinguish the two kinds go missing from the rendered chart templates.

Out of scope: CEL semantics against the in-memory builders, covered by
``tests/unit/kubernetes/test_crd_validation_adversarial.py``.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATES_DIR = REPO_ROOT / "deploy" / "helm" / "aiperf-operator" / "templates"
JOB_CRD = TEMPLATES_DIR / "crd-aiperfjob.yaml"
SWEEP_CRD = TEMPLATES_DIR / "crd-aiperfsweep.yaml"


def test_generated_crds_are_current() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tools.generate_crd", "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        "CRDs are stale. Run: uv run python tools/generate_crd.py\n"
        f"{result.stdout}\n{result.stderr}"
    )


@pytest.mark.parametrize(
    "crd_path,expected_rule,forbidden_rule",
    [
        pytest.param(
            JOB_CRD, "- rule: '!has(self.sweep)'", "- rule: has(self.sweep)",
            id="aiperfjob-forbids-sweep",
        ),
        pytest.param(
            SWEEP_CRD, "- rule: has(self.sweep)", "- rule: '!has(self.sweep)'",
            id="aiperfsweep-requires-sweep",
        ),
    ],
)  # fmt: skip
def test_kind_specific_sweep_cel_rule_in_rendered_template(
    crd_path: Path, expected_rule: str, forbidden_rule: str
) -> None:
    text = crd_path.read_text()
    assert expected_rule in text, (
        f"{crd_path.name} is missing its kind-specific spec.sweep CEL rule "
        f"{expected_rule!r}; regenerate with tools/generate_crd.py"
    )
    assert forbidden_rule not in text, (
        f"{crd_path.name} carries the opposite kind's spec.sweep CEL rule "
        f"{forbidden_rule!r}"
    )
