# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for --profile-export-prefix applying to the outputs.json export.

outputs.json used to be the one export path that ignored the prefix, so a prefix
resolving to 'outputs' collided with the summary JSON and was rejected outright.
It now takes the prefix like every other artifact (`foo` -> `foo_outputs.json`),
which makes the collision impossible and the guard unnecessary.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.artifacts import ArtifactsConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


class TestOutputsJsonPrefix:
    """ArtifactsConfig.outputs_json_file honors --profile-export-prefix."""

    def test_outputs_json_file_no_prefix_uses_historical_name(self) -> None:
        cfg = ArtifactsConfig()
        assert cfg.outputs_json_file.name == "outputs.json"

    def test_outputs_json_file_with_prefix_is_prefixed(self) -> None:
        cfg = ArtifactsConfig(prefix="foo")
        assert cfg.outputs_json_file.name == "foo_outputs.json"

    @pytest.mark.parametrize(
        "prefix",
        [
            param("foo", id="bare"),
            param("foo.json", id="with-json-suffix"),
            param("foo.jsonl", id="with-jsonl-suffix"),
            param("foo_raw.jsonl", id="with-raw-suffix"),
            param("foo_outputs.json", id="with-outputs-suffix"),
            param("foo_timeslices.json", id="with-timeslices-suffix"),
        ],
    )  # fmt: skip
    def test_outputs_json_file_strips_known_suffixes(self, prefix: str) -> None:
        cfg = ArtifactsConfig(prefix=prefix)
        assert cfg.outputs_json_file.name == "foo_outputs.json"

    def test_outputs_suffix_strips_before_json_suffix(self) -> None:
        """`_outputs.json` must win over `.json` (longest match first)."""
        cfg = ArtifactsConfig(prefix="foo_outputs.json")
        assert cfg._base() == "foo"

    @pytest.mark.parametrize(
        "prefix",
        [
            param("outputs", id="bare"),
            param("outputs.json", id="with-json-suffix"),
            param("outputs_raw.jsonl", id="with-raw-suffix"),
        ],
    )  # fmt: skip
    def test_formerly_colliding_prefix_now_resolves_distinctly(
        self, prefix: str
    ) -> None:
        """A prefix of 'outputs' used to be a hard error; the paths no longer collide."""
        cfg = ArtifactsConfig(prefix=prefix, export_outputs_json=True)
        assert cfg.profile_export_json_file.name == "outputs.json"
        assert cfg.outputs_json_file.name == "outputs_outputs.json"
        assert cfg.profile_export_json_file != cfg.outputs_json_file

    def test_outputs_json_file_distinct_from_every_other_export_path(self) -> None:
        cfg = ArtifactsConfig(prefix="foo", export_outputs_json=True)
        others = {
            cfg.profile_export_json_file,
            cfg.profile_export_csv_file,
            cfg.profile_export_jsonl_file,
            cfg.profile_export_raw_jsonl_file,
            cfg.profile_export_timeslices_json_file,
        }
        assert cfg.outputs_json_file not in others


class TestOutputsJsonPrefixCLI:
    """End-to-end: the CLI converter carries the prefix into outputs.json."""

    def test_convert_cli_to_aiperf_prefixes_outputs_json(self) -> None:
        cli = CLIConfig(
            model_names=["m"],
            profile_export_prefix="foo",
            export_outputs_json=True,
        )
        cfg = convert_cli_to_aiperf(cli)
        assert cfg.benchmark.artifacts.outputs_json_file.name == "foo_outputs.json"

    def test_convert_cli_to_aiperf_formerly_colliding_prefix_allowed(self) -> None:
        cli = CLIConfig(
            model_names=["m"],
            profile_export_prefix="outputs",
            export_outputs_json=True,
        )
        cfg = convert_cli_to_aiperf(cli)
        artifacts = cfg.benchmark.artifacts
        assert artifacts.profile_export_json_file.name == "outputs.json"
        assert artifacts.outputs_json_file.name == "outputs_outputs.json"
