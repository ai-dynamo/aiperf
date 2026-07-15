# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for kube generate/validate kind dispatch.

Focuses on:
- AIPerfJob versus AIPerfSweep dispatch from config shape and CR ``kind``.
- Trust-boundary YAML cases: multi-document streams, missing kind, unknown kind.
- JSON/text output surfaces preserving path, kind, and ``spec.sweep`` context.
- Namespace propagation for generated CRs without leaking kube client-only options.

Out of scope: cluster submission, kubectl invocation, and raw JobSet manifests; see
``tests/unit/cli_commands/kube/test_profile_deploy.py`` and
``tests/unit/cli_commands/kube/test_profile_deploy_direct.py`` for those paths.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

import orjson
import pytest
import ruamel.yaml
import yaml
from pytest import param

from aiperf.cli_commands.kube import generate as generate_cmd
from aiperf.config.flags import CLIConfig
from aiperf.config.kube import KubeOptions
from aiperf.kubernetes import validate as kube_validate
from aiperf.operator.models import AIPerfJobSpec, AIPerfSweepSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_VALID_BENCHMARK: dict[str, object] = {
    "models": ["meta-llama/Llama-3.1-8B-Instruct"],
    "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
    "datasets": [
        {
            "name": "main",
            "type": "synthetic",
            "entries": 16,
            "prompts": {"isl": 64, "osl": 32},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 8,
            "concurrency": 2,
        }
    ],
}

_GRID_SWEEP: dict[str, object] = {
    "type": "grid",
    "variables": {"phases.profiling.concurrency": [1, 2]},
}


def _benchmark_with(**overrides: object) -> dict[str, object]:
    """Return a real benchmark payload with one adversarial override applied."""
    benchmark = copy.deepcopy(_VALID_BENCHMARK)
    benchmark.update(overrides)
    return benchmark


def _envelope_with(**overrides: object) -> dict[str, object]:
    """Return a real AIPerfConfig envelope for generate-path validation."""
    envelope: dict[str, object] = {"benchmark": _benchmark_with()}
    envelope.update(overrides)
    return envelope


def _job_doc_with(**overrides: object) -> dict[str, object]:
    """Return a valid AIPerfJob CR document with caller-provided overrides."""
    doc: dict[str, object] = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": "llama-baseline"},
        "spec": {"benchmark": _benchmark_with(), "image": "aiperf:ci"},
    }
    doc.update(overrides)
    return doc


def _sweep_doc_with(**overrides: object) -> dict[str, object]:
    """Return a valid AIPerfSweep CR document with caller-provided overrides."""
    doc = _job_doc_with(
        kind="AIPerfSweep",
        metadata={"name": "llama-concurrency-sweep"},
    )
    spec = dict(doc["spec"])
    spec["sweep"] = copy.deepcopy(_GRID_SWEEP)
    doc["spec"] = spec
    doc.update(overrides)
    return doc


def _write_yaml(path: Path, doc: dict[str, object]) -> Path:
    """Write ``doc`` to ``path`` and return the same path for CLI calls."""
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _write_config(path: Path, envelope: dict[str, object]) -> Path:
    """Write an AIPerfConfig envelope used by ``aiperf kube generate``."""
    return _write_yaml(path, envelope)


def _cli_config(config_file: Path) -> CLIConfig:
    """Return CLI input with only ``--config`` explicitly set."""
    return CLIConfig(config_file=config_file)


def _kube_options(**overrides: object) -> KubeOptions:
    """Return realistic kube deployment options for generate-path tests."""
    data: dict[str, object] = {"image": "nvcr.io/nvidia/aiperf:ci"}
    data.update(overrides)
    return KubeOptions.model_validate(data)


def _parse_single_yaml(stdout: str) -> dict[str, object]:
    """Parse a single YAML document emitted by ``aiperf kube generate``."""
    parsed = ruamel.yaml.YAML(typ="safe").load(stdout)
    assert isinstance(parsed, dict)
    return parsed


def _joined_errors(result: kube_validate.ValidationResult) -> str:
    """Join validation errors for regex-free multi-keyword assertions."""
    return "\n".join(result.errors)


# ---------------------------------------------------------------------------
# Generate kind dispatch
# ---------------------------------------------------------------------------


class TestGenerateKindDispatch:
    """``aiperf kube generate`` chooses CR kind from the resolved config envelope."""

    def test_resolve_spec_and_name_without_sweep_selects_aiperfjob_spec(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(tmp_path / "llama_baseline.yaml", _envelope_with())

        spec, config, name = generate_cmd._resolve_spec_and_name(
            _cli_config(config_file),
            _kube_options(name="llama-baseline"),
        )

        assert generate_cmd._choose_kind(config) == "AIPerfJob"
        assert name == "llama-baseline"
        assert "sweep" not in spec
        validated = AIPerfJobSpec.model_validate(spec)
        assert validated.benchmark.endpoint.urls == ["http://localhost:8000"]

    def test_resolve_spec_and_name_with_sweep_selects_aiperfsweep_spec(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(
            tmp_path / "llama_grid.yaml",
            _envelope_with(sweep=copy.deepcopy(_GRID_SWEEP)),
        )

        spec, config, name = generate_cmd._resolve_spec_and_name(
            _cli_config(config_file),
            _kube_options(name="llama-concurrency-sweep"),
        )

        assert generate_cmd._choose_kind(config) == "AIPerfSweep"
        assert name == "llama-concurrency-sweep"
        assert spec["sweep"]["type"] == "grid"
        assert "sweep" not in spec["benchmark"]
        validated = AIPerfSweepSpec.model_validate(spec)
        assert validated.sweep is not None

    @pytest.mark.asyncio
    async def test_generate_operator_with_sweep_outputs_aiperfsweep_cr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_file = _write_config(
            tmp_path / "operator_sweep.yaml",
            _envelope_with(sweep=copy.deepcopy(_GRID_SWEEP)),
        )

        with patch("aiperf.cli_commands.kube.generate._print_memory_estimate"):
            await generate_cmd.generate(
                cli_config=_cli_config(config_file),
                kube_options=_kube_options(
                    name="operator-sweep",
                    namespace="aiperf-ci",
                    kube_context="kind-aiperf-ci",
                ),
                operator=True,
            )

        cr = _parse_single_yaml(capsys.readouterr().out)
        assert cr["kind"] == "AIPerfSweep"
        assert cr["metadata"] == {"name": "operator-sweep", "namespace": "aiperf-ci"}
        assert cr["spec"]["sweep"]["variables"] == {
            "phases.profiling.concurrency": [1, 2]
        }
        assert "kubeContext" not in cr["spec"]
        assert "kube_context" not in cr["spec"]

    @pytest.mark.asyncio
    async def test_generate_operator_without_sweep_outputs_aiperfjob_cr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_file = _write_config(tmp_path / "operator_job.yaml", _envelope_with())

        with patch("aiperf.cli_commands.kube.generate._print_memory_estimate"):
            await generate_cmd.generate(
                cli_config=_cli_config(config_file),
                kube_options=_kube_options(
                    name="operator-job",
                    namespace="aiperf-ci",
                ),
                operator=True,
            )

        cr = _parse_single_yaml(capsys.readouterr().out)
        assert cr["kind"] == "AIPerfJob"
        assert cr["metadata"] == {"name": "operator-job", "namespace": "aiperf-ci"}
        assert "sweep" not in cr["spec"]


# ---------------------------------------------------------------------------
# Validate YAML trust boundary
# ---------------------------------------------------------------------------


class TestValidateYamlTrustBoundary:
    """Malformed CR envelope shapes fail with field-specific context."""

    @pytest.mark.parametrize(
        "kind",
        [
            param(None, id="missing-kind"),
            param("AIPerfWorkflow", id="unknown-kind"),
            param("aiperfjob", id="lowercase-kind"),
        ],
    )  # fmt: skip
    def test_validate_file_missing_or_unknown_kind_names_expected_kinds(
        self, tmp_path: Path, kind: str | None
    ) -> None:
        doc = _job_doc_with()
        if kind is None:
            del doc["kind"]
        else:
            doc["kind"] = kind
        path = _write_yaml(tmp_path / "bad-kind.yaml", doc)

        result = kube_validate.validate_file(path)

        assert result.passed is False
        errors = _joined_errors(result)
        assert "kind" in errors
        assert "AIPerfJob" in errors
        assert "AIPerfSweep" in errors
        assert str(kind) in errors

    def test_validate_file_multidoc_stream_fails_instead_of_ignoring_second_doc(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "two-docs.yaml"
        path.write_text(
            yaml.safe_dump(_job_doc_with(), sort_keys=False)
            + "---\n"
            + yaml.safe_dump(
                _job_doc_with(kind="AIPerfWorkflow", metadata={"name": "bad-kind"}),
                sort_keys=False,
            )
        )

        result = kube_validate.validate_file(path)

        assert result.passed is False
        errors = _joined_errors(result)
        assert "YAML parse error" in errors
        assert "expected a single document" in errors

    @pytest.mark.parametrize(
        "kind,sweep_value,expected",
        [
            param(
                "AIPerfJob",
                copy.deepcopy(_GRID_SWEEP),
                "must be null on AIPerfJob",
                id="job-rejects-sweep",
            ),
            param(
                "AIPerfSweep",
                None,
                "required on AIPerfSweep",
                id="sweep-requires-sweep",
            ),
            param(
                "AIPerfSweep",
                {},
                "non-empty sweep mapping",
                id="sweep-rejects-empty-mapping",
            ),
        ],
    )  # fmt: skip
    def test_validate_file_kind_sweep_cardinality_errors_name_kind_and_field(
        self,
        tmp_path: Path,
        kind: str,
        sweep_value: dict[str, object] | None,
        expected: str,
    ) -> None:
        doc = _job_doc_with(kind=kind, metadata={"name": "cardinality-lock"})
        spec = dict(doc["spec"])
        if sweep_value is not None:
            spec["sweep"] = sweep_value
        doc["spec"] = spec
        path = _write_yaml(tmp_path / "cardinality.yaml", doc)

        result = kube_validate.validate_file(path)

        assert result.passed is False
        errors = _joined_errors(result)
        assert "spec.sweep" in errors
        assert expected in errors

    def test_validate_file_valid_aiperfsweep_dispatches_to_sweep_schema(
        self, tmp_path: Path
    ) -> None:
        path = _write_yaml(tmp_path / "valid-sweep.yaml", _sweep_doc_with())

        result = kube_validate.validate_file(path)

        assert result.passed is True
        assert result.errors == []


# ---------------------------------------------------------------------------
# CLI output surfaces
# ---------------------------------------------------------------------------


class TestValidateCliOutputSurfaces:
    """CLI wrappers preserve machine-readable and human-readable error context."""

    @pytest.mark.asyncio
    async def test_validate_json_output_reports_each_file_path_and_kind_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        good_path = _write_yaml(tmp_path / "good-job.yaml", _job_doc_with())
        bad_path = _write_yaml(
            tmp_path / "bad-kind.yaml",
            _job_doc_with(kind="AIPerfWorkflow", metadata={"name": "bad-kind"}),
        )

        with pytest.raises(SystemExit) as exc_info:
            from aiperf.cli_commands.kube.validate import validate

            await validate(files=[good_path, bad_path], output="json")

        assert exc_info.value.code == 1
        payload = orjson.loads(capsys.readouterr().out)
        assert [entry["path"] for entry in payload] == [str(good_path), str(bad_path)]
        assert payload[0]["passed"] is True
        assert payload[1]["passed"] is False
        assert any("AIPerfWorkflow" in error for error in payload[1]["errors"])

    @pytest.mark.asyncio
    async def test_validate_text_output_reports_sweep_cardinality_context(
        self, tmp_path: Path
    ) -> None:
        doc = _job_doc_with()
        spec = dict(doc["spec"])
        spec["sweep"] = copy.deepcopy(_GRID_SWEEP)
        doc["spec"] = spec
        path = _write_yaml(tmp_path / "job-with-sweep.yaml", doc)

        with (
            patch("aiperf.kubernetes.validate.kube_console.print_error") as print_error,
            patch("aiperf.kubernetes.validate.kube_console.logger.info") as log_info,
            pytest.raises(SystemExit) as exc_info,
        ):
            from aiperf.cli_commands.kube.validate import validate

            await validate(files=[path])

        assert exc_info.value.code == 1
        assert any(
            str(path) in str(call.args[0]) for call in print_error.call_args_list
        )
        rendered = "\n".join(str(call.args[0]) for call in log_info.call_args_list)
        assert "spec.sweep" in rendered
        assert "AIPerfJob" in rendered
        assert "AIPerfSweep" in rendered
