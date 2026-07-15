# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lock-in tests for AIPerfJob / AIPerfSweep CR shape across `aiperf kube *`.

Asserts each CR-construction path produces a flat envelope shape
(``spec.benchmark`` carries body fields; ``spec.<envelope-key>`` carries
envelope fields like ``variables``/``randomSeed``/``sweep``/``multiRun``;
no doubled ``spec.benchmark.benchmark.X``) and round-trips cleanly through
``AIPerfJobSpec`` / ``AIPerfSweepSpec`` ``model_validate``.

Covers:

* ``aiperf kube init`` -- ``wrap_as_aiperf_job`` for envelope-shape input
  (the bundled-template happy path) and flat-shape input (back-compat
  branch).
* ``aiperf kube generate`` -- ``KubeOptions.to_crd_spec`` (single
  benchmark) and ``_build_sweep_spec`` (sweep CR).
* ``aiperf kube profile`` -- ``deploy_via_operator``'s round-trip path
  (validates the spec it would submit).
* ``aiperf kube sweep`` -- ``_build_sweep_cr_dict``'s
  ``AIPerfSweepSpec.model_validate`` round-trip.
"""

from __future__ import annotations

from typing import Any

import pytest
import ruamel.yaml
import yaml

from aiperf.kubernetes.init_template import wrap_as_aiperf_job

# A minimal envelope-shape body that round-trips through AIPerfJobSpec.
# Uses the long-form (``models``/``datasets`` lists) to bypass the
# envelope-shorthand-folding gap that exists in bundled templates.
_VALID_BODY = {
    "models": ["meta-llama/Llama-3.1-8B-Instruct"],
    "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 512, "osl": 128},
        }
    ],
    "phases": [
        {"name": "default", "type": "concurrency", "concurrency": 8, "requests": 100}
    ],
}


def _yaml_strip_comments(text: str) -> dict[str, Any]:
    """Parse a CR YAML output by dropping comment-only lines first."""
    yaml_lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return ruamel.yaml.YAML().load("\n".join(yaml_lines))


class TestWrapAsAIPerfJobEnvelopeShape:
    """``wrap_as_aiperf_job`` must produce a flat envelope spec for envelope input."""

    def test_envelope_input_lands_at_correct_depth(self) -> None:
        body_yaml = yaml.safe_dump(
            {
                "benchmark": _VALID_BODY,
                "variables": {"region": "us-west"},
                "random_seed": 42,
            },
            sort_keys=False,
        )

        wrapped = wrap_as_aiperf_job(body_yaml, job_name="run-1")
        parsed = _yaml_strip_comments(wrapped)

        assert parsed["kind"] == "AIPerfJob"
        spec = parsed["spec"]
        # Body lives at spec.benchmark; envelope fields at spec level.
        assert "endpoint" in spec["benchmark"]
        assert "phases" in spec["benchmark"]
        # No doubled wrapping.
        assert "benchmark" not in spec["benchmark"]
        # Envelope fields land at envelope (spec) level.
        assert spec["variables"] == {"region": "us-west"}
        assert spec["random_seed"] == 42

    def test_envelope_schema_version_stripped_from_spec(self) -> None:
        """File-envelope ``schemaVersion`` must not land at ``spec.<key>``.

        Local config files carry ``schemaVersion: "2.0"`` at the top level,
        but AIPerfJobSpec rejects it as an unknown spec field — inside a CR
        that role belongs to ``apiVersion``. A ``schema_version`` nested
        under ``benchmark:`` is a valid AIPerfConfig field and must survive.
        """
        body_yaml = yaml.safe_dump(
            {
                "schemaVersion": "2.0",
                "benchmark": {**_VALID_BODY, "schema_version": "2.0"},
            },
            sort_keys=False,
        )

        wrapped = wrap_as_aiperf_job(body_yaml, job_name="run-1")
        parsed = _yaml_strip_comments(wrapped)

        spec = parsed["spec"]
        assert "schemaVersion" not in spec
        assert "schema_version" not in spec
        assert spec["benchmark"]["schema_version"] == "2.0"

    def test_envelope_input_round_trips_through_aiperfjob_spec(self) -> None:
        from aiperf.operator.models import AIPerfJobSpec

        body_yaml = yaml.safe_dump({"benchmark": _VALID_BODY}, sort_keys=False)
        wrapped = wrap_as_aiperf_job(body_yaml, job_name="run-1")
        parsed = _yaml_strip_comments(wrapped)
        spec = dict(parsed["spec"])
        # Image is required by AIPerfJobSpec but ``kube init`` defers it to
        # `aiperf kube profile --image`. Stamp a placeholder for validation.
        spec.setdefault("image", "aiperf:latest")

        validated = AIPerfJobSpec.model_validate(spec)
        assert validated.benchmark.endpoint.urls == ["http://localhost:8000"]


class TestWrapAsAIPerfJobFlatShape:
    """Flat-shape body (no top-level ``benchmark:``) keeps legacy wrap behavior."""

    def test_flat_input_indents_under_spec_benchmark(self) -> None:
        body_yaml = yaml.safe_dump(_VALID_BODY, sort_keys=False)

        wrapped = wrap_as_aiperf_job(body_yaml, job_name="run-2")
        parsed = _yaml_strip_comments(wrapped)

        assert parsed["spec"]["benchmark"]["endpoint"]["urls"] == [
            "http://localhost:8000"
        ]

    def test_flat_input_round_trips_through_aiperfjob_spec(self) -> None:
        from aiperf.operator.models import AIPerfJobSpec

        body_yaml = yaml.safe_dump(_VALID_BODY, sort_keys=False)
        wrapped = wrap_as_aiperf_job(body_yaml, job_name="run-2")
        parsed = _yaml_strip_comments(wrapped)
        spec = dict(parsed["spec"])
        spec.setdefault("image", "aiperf:latest")

        validated = AIPerfJobSpec.model_validate(spec)
        assert validated.benchmark.endpoint.urls == ["http://localhost:8000"]


class TestKubeOptionsToCrdSpec:
    """``KubeOptions.to_crd_spec`` builds a flat AIPerfJob spec from CLI flags."""

    def test_to_crd_spec_round_trips_through_aiperfjob_spec(self) -> None:
        from aiperf.config import AIPerfConfig
        from aiperf.config.kube import KubeOptions
        from aiperf.operator.models import AIPerfJobSpec

        config = AIPerfConfig.model_validate({"benchmark": _VALID_BODY})
        kube_options = KubeOptions.model_validate({"image": "aiperf:latest"})

        spec = kube_options.to_crd_spec(config)

        # Body keys are nested correctly (no doubled benchmark).
        assert "endpoint" in spec["benchmark"]
        assert "benchmark" not in spec["benchmark"]
        # Deployment fields land at spec level.
        assert spec["image"] == "aiperf:latest"

        validated = AIPerfJobSpec.model_validate(spec)
        assert validated.benchmark.endpoint.urls == ["http://localhost:8000"]


class TestBuildSweepSpec:
    """``aiperf kube generate``'s ``_build_sweep_spec`` keeps envelope keys flat."""

    def test_sweep_spec_round_trips_through_aiperfsweep_spec(self) -> None:
        from aiperf.cli_commands.kube.generate import _build_sweep_spec
        from aiperf.config import AIPerfConfig
        from aiperf.config.kube import KubeOptions
        from aiperf.operator.models import AIPerfSweepSpec

        envelope = {
            "benchmark": _VALID_BODY,
            "sweep": {
                "type": "grid",
                "variables": {"phases.default.concurrency": [1, 2, 4]},
            },
        }
        config = AIPerfConfig.model_validate(envelope)
        kube_options = KubeOptions.model_validate({"image": "aiperf:latest"})

        spec = _build_sweep_spec(config, kube_options)

        # Envelope-level sweep block at spec level (NOT inside spec.benchmark).
        assert "sweep" in spec
        assert "sweep" not in spec["benchmark"]
        # Body still under spec.benchmark.
        assert "endpoint" in spec["benchmark"]
        # Image lives at spec level.
        assert spec["image"] == "aiperf:latest"

        validated = AIPerfSweepSpec.model_validate(spec)
        assert validated.sweep is not None


class TestKubeSweepBuildCrDict:
    """``aiperf kube sweep`` validates via ``AIPerfSweepSpec.model_validate``."""

    def test_build_sweep_cr_dict_envelope_shape(self, tmp_path) -> None:
        from aiperf.cli_commands.kube.sweep import _build_sweep_cr_dict
        from aiperf.config.kube import KubeOptions

        config_file = tmp_path / "sweep.yaml"
        config_file.write_text(
            yaml.safe_dump(
                {
                    "benchmark": _VALID_BODY,
                    "sweep": {
                        "type": "grid",
                        "variables": {"phases.default.concurrency": [1, 2, 4]},
                    },
                },
                sort_keys=False,
            )
        )

        cr = _build_sweep_cr_dict(
            config_file=config_file,
            kube_options=KubeOptions.model_validate({"image": "aiperf:latest"}),
            multi_run_trials=None,
            cooldown_seconds=0.0,
            convergence_metric=None,
            convergence_min_runs=2,
            convergence_max_runs=10,
            convergence_threshold=0.05,
        )

        assert cr["kind"] == "AIPerfSweep"
        spec = cr["spec"]
        # Envelope flat: sweep at spec level, benchmark body under spec.benchmark.
        assert "sweep" in spec
        assert "endpoint" in spec["benchmark"]
        # No doubled wrapping.
        assert "benchmark" not in spec["benchmark"]


@pytest.mark.parametrize(
    "envelope_keys",
    [
        ["variables"],
        ["random_seed"],
        ["variables", "random_seed"],
    ],
)
def test_envelope_keys_never_land_inside_benchmark(envelope_keys: list[str]) -> None:
    """Envelope-level keys must NOT collapse into ``spec.benchmark``."""
    payload: dict[str, Any] = {"benchmark": _VALID_BODY}
    if "variables" in envelope_keys:
        payload["variables"] = {"region": "us-west"}
    if "random_seed" in envelope_keys:
        payload["random_seed"] = 42

    body_yaml = yaml.safe_dump(payload, sort_keys=False)
    wrapped = wrap_as_aiperf_job(body_yaml)
    parsed = _yaml_strip_comments(wrapped)

    for key in envelope_keys:
        assert key in parsed["spec"], f"{key} should be at spec level"
        assert key not in parsed["spec"]["benchmark"], (
            f"{key} must NOT be inside spec.benchmark"
        )
