# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `--export-level raw` implying `--export-outputs-json`.

Asking for the full request/response bodies should also give you the generated
text. The implication is a default, not a lock: an explicit
`--no-export-outputs-json` (or `exportOutputsJson: false` in YAML) still wins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiperf.common.enums import ExportFormat, ExportLevel
from aiperf.config.artifacts import ArtifactsConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.flags.resolver import resolve_config


def _artifacts(**kwargs: Any) -> ArtifactsConfig:
    return ArtifactsConfig(records=[ExportFormat.JSONL], **kwargs)


class TestRawImpliesOutputsJson:
    """ArtifactsConfig resolves the implication during validation."""

    def test_raw_enables_outputs_json(self) -> None:
        assert _artifacts(raw=True).export_outputs_json is True

    def test_raw_with_explicit_false_stays_disabled(self) -> None:
        cfg = _artifacts(raw=True, export_outputs_json=False)
        assert cfg.export_outputs_json is False

    def test_records_level_does_not_enable_outputs_json(self) -> None:
        assert _artifacts(raw=False).export_outputs_json is False

    def test_summary_level_does_not_enable_outputs_json(self) -> None:
        cfg = ArtifactsConfig(records=False, raw=False)
        assert cfg.export_outputs_json is False

    def test_flag_alone_works_below_raw(self) -> None:
        """The flag is still independently usable; summary + text is a valid state."""
        cfg = ArtifactsConfig(records=False, raw=False, export_outputs_json=True)
        assert cfg.export_level == ExportLevel.SUMMARY
        assert cfg.export_outputs_json is True


class TestRawImpliesOutputsJsonCLI:
    """End-to-end through the CLI converter."""

    def test_export_level_raw_enables_outputs_json(self) -> None:
        cli = CLIConfig(model_names=["m"], export_level=ExportLevel.RAW)
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.benchmark.artifacts.export_outputs_json is True

    def test_no_export_outputs_json_opts_out_of_the_implication(self) -> None:
        cli = CLIConfig(
            model_names=["m"],
            export_level=ExportLevel.RAW,
            export_outputs_json=False,
        )
        cfg = convert_cli_to_aiperf(cli)
        artifacts = cfg.benchmark.artifacts
        assert artifacts.export_level == ExportLevel.RAW
        assert artifacts.export_outputs_json is False

    def test_export_level_records_leaves_outputs_json_off(self) -> None:
        cli = CLIConfig(model_names=["m"], export_level=ExportLevel.RECORDS)
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.benchmark.artifacts.export_outputs_json is False


class TestRawImpliesOutputsJsonUnderConfigYaml:
    """The implication through `resolve_config` with a real `--config` YAML.

    The `--config` layer is the one the repo's flag-routing invariant singles
    out, and the implication depends on a contract no other test states:
    `build_artifacts` propagates `export_outputs_json` only when the user set
    it, so `model_fields_set` stays empty otherwise. A refactor that always
    emitted the field would silently kill the implication.
    """

    @staticmethod
    def _yaml(tmp_path: Path, artifacts: str) -> Path:
        cfg = tmp_path / "base.yaml"
        cfg.write_text(
            'schemaVersion: "2.0"\n'
            "benchmark:\n"
            "  model: test-model\n"
            "  endpoint:\n"
            "    url: http://localhost:8000\n"
            "  dataset:\n"
            "    type: synthetic\n"
            "  phases:\n"
            "    type: concurrency\n"
            "    concurrency: 1\n"
            "    requests: 5\n"
            f"  artifacts:\n{artifacts}"
        )
        return cfg

    @staticmethod
    def _cli(**kwargs: Any) -> CLIConfig:
        """A CLIConfig whose model_fields_set is exactly ``kwargs``."""
        return CLIConfig(**CLIConfig(**kwargs).model_dump(exclude_unset=True))

    def test_yaml_raw_true_implies_outputs_json(self, tmp_path: Path) -> None:
        cfg = resolve_config(self._cli(), self._yaml(tmp_path, "    raw: true\n"))
        assert cfg.benchmark.artifacts.export_outputs_json is True

    def test_yaml_explicit_false_survives_cli_export_level_raw(
        self, tmp_path: Path
    ) -> None:
        cfg = resolve_config(
            self._cli(export_level=ExportLevel.RAW),
            self._yaml(tmp_path, "    exportOutputsJson: false\n"),
        )
        artifacts = cfg.benchmark.artifacts
        assert artifacts.export_level == ExportLevel.RAW
        assert artifacts.export_outputs_json is False

    def test_yaml_raw_true_with_cli_opt_out(self, tmp_path: Path) -> None:
        cfg = resolve_config(
            self._cli(export_outputs_json=False),
            self._yaml(tmp_path, "    raw: true\n"),
        )
        artifacts = cfg.benchmark.artifacts
        assert artifacts.export_level == ExportLevel.RAW
        assert artifacts.export_outputs_json is False
