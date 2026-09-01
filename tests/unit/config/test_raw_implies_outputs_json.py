# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `--export-level raw` implying `--export-outputs-json`.

Asking for the full request/response bodies should also give you the generated
text. The implication is a default, not a lock: an explicit
`--no-export-outputs-json` (or `exportOutputsJson: false` in YAML) still wins.
"""

from __future__ import annotations

from aiperf.common.enums import ExportFormat, ExportLevel
from aiperf.config.artifacts import ArtifactsConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


def _artifacts(**kwargs) -> ArtifactsConfig:
    return ArtifactsConfig(records=[ExportFormat.JSONL], **kwargs)


class TestRawImpliesOutputsJson:
    """ArtifactsConfig resolves the implication during validation."""

    def test_raw_enables_outputs_json(self) -> None:
        assert _artifacts(raw=True).export_outputs_json is True

    def test_raw_with_explicit_false_stays_disabled(self) -> None:
        cfg = _artifacts(raw=True, export_outputs_json=False)
        assert cfg.export_outputs_json is False

    def test_raw_with_explicit_true_stays_enabled(self) -> None:
        cfg = _artifacts(raw=True, export_outputs_json=True)
        assert cfg.export_outputs_json is True

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

    def test_raw_outputs_json_honors_prefix(self) -> None:
        """The implied file still takes the prefix, like every other artifact."""
        cli = CLIConfig(
            model_names=["m"],
            export_level=ExportLevel.RAW,
            profile_export_prefix="foo",
        )
        cfg = convert_cli_to_aiperf(cli)
        artifacts = cfg.benchmark.artifacts
        assert artifacts.export_outputs_json is True
        assert artifacts.outputs_json_file.name == "foo_outputs.json"
