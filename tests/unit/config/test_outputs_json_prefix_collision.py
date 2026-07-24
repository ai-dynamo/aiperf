# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for --profile-export-prefix / --export-outputs-json path collision guard.

When both flags are active and the prefix resolves to 'outputs', the summary
JSON (`outputs.json`) and the generated-output export (`outputs.json`) would
target the same file. The model validator must reject this combination.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.artifacts import ArtifactsConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


class TestOutputsJsonPrefixCollision:
    """ArtifactsConfig.validate_artifacts rejects colliding prefix + export_outputs_json."""

    @pytest.mark.parametrize(
        "prefix",
        [
            param("outputs", id="bare"),
            param("outputs.json", id="with-json-suffix"),
            param("outputs.jsonl", id="with-jsonl-suffix"),
            param("outputs.csv", id="with-csv-suffix"),
            param("outputs_raw.jsonl", id="with-raw-suffix"),
            param("outputs_timeslices.json", id="with-timeslices-suffix"),
        ],
    )  # fmt: skip
    def test_validate_artifacts_colliding_prefix_raises_error(
        self, prefix: str
    ) -> None:
        with pytest.raises(
            ValidationError, match="colliding with --export-outputs-json"
        ):
            ArtifactsConfig(prefix=prefix, export_outputs_json=True)

    def test_validate_artifacts_colliding_prefix_without_export_allows(self) -> None:
        cfg = ArtifactsConfig(prefix="outputs", export_outputs_json=False)
        assert cfg.profile_export_json_file.name == "outputs.json"

    def test_validate_artifacts_non_colliding_prefix_with_export_allows(self) -> None:
        cfg = ArtifactsConfig(prefix="foo", export_outputs_json=True)
        assert cfg.profile_export_json_file.name == "foo.json"
        assert cfg.outputs_json_file.name == "outputs.json"

    def test_validate_artifacts_no_prefix_with_export_allows(self) -> None:
        cfg = ArtifactsConfig(export_outputs_json=True)
        assert cfg.profile_export_json_file.name == "profile_export_aiperf.json"
        assert cfg.outputs_json_file.name == "outputs.json"


class TestOutputsJsonPrefixCollisionCLI:
    """End-to-end: CLI converter propagates the collision to ArtifactsConfig validation."""

    def test_convert_cli_to_aiperf_colliding_prefix_with_export_raises_error(
        self,
    ) -> None:
        cli = CLIConfig(
            model_names=["m"],
            profile_export_prefix="outputs",
            export_outputs_json=True,
        )
        with pytest.raises(
            ValidationError, match="colliding with --export-outputs-json"
        ):
            convert_cli_to_aiperf(cli)

    def test_convert_cli_to_aiperf_colliding_prefix_without_export_allows(self) -> None:
        cli = CLIConfig(model_names=["m"], profile_export_prefix="outputs")
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.benchmark.artifacts.profile_export_json_file.name == "outputs.json"
