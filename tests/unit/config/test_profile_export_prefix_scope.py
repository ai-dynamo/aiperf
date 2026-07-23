# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for `--profile-export-prefix`."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.artifacts import ArtifactsConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


class TestPrefixAppliedToEveryExport:
    """When prefix is set, EVERY exported file uses `<prefix>...` as its base."""

    def _cfg(self) -> ArtifactsConfig:
        return ArtifactsConfig(prefix="foo")

    @pytest.mark.parametrize(
        "attr, expected",
        [
            param("profile_export_csv_file", "foo.csv", id="csv_summary"),
            param("profile_export_json_file", "foo.json", id="json_summary"),
            param("profile_export_timeslices_csv_file", "foo_timeslices.csv", id="timeslices_csv"),
            param("profile_export_timeslices_json_file", "foo_timeslices.json", id="timeslices_json"),
            param("profile_export_jsonl_file", "foo.jsonl", id="per_record_jsonl"),
            param("profile_export_raw_jsonl_file", "foo_raw.jsonl", id="raw_jsonl"),
            param("profile_export_gpu_telemetry_jsonl_file", "foo_gpu_telemetry.jsonl", id="gpu_telemetry_jsonl"),
            param("server_metrics_export_jsonl_file", "foo_server_metrics.jsonl", id="server_metrics_jsonl"),
            param("server_metrics_export_json_file", "foo_server_metrics.json", id="server_metrics_json"),
            param("server_metrics_export_csv_file", "foo_server_metrics.csv", id="server_metrics_csv"),
            param("server_metrics_export_parquet_file", "foo_server_metrics.parquet", id="server_metrics_parquet"),
        ],
    )  # fmt: skip
    def test_prefixed_export_name(self, attr, expected):
        assert getattr(self._cfg(), attr).name == expected


class TestUnsetPrefixPreservesPerFileDefaults:
    """When prefix is not set, historical per-file default names are used."""

    def _cfg(self) -> ArtifactsConfig:
        return ArtifactsConfig()

    @pytest.mark.parametrize(
        "attr, expected",
        [
            param("profile_export_csv_file", "profile_export_aiperf.csv", id="csv_summary_default"),
            param("profile_export_json_file", "profile_export_aiperf.json", id="json_summary_default"),
            param("profile_export_timeslices_csv_file", "profile_export_aiperf_timeslices.csv", id="timeslices_csv_default"),
            param("profile_export_timeslices_json_file", "profile_export_aiperf_timeslices.json", id="timeslices_json_default"),
            param("profile_export_jsonl_file", "profile_export.jsonl", id="per_record_jsonl_default"),
            param("profile_export_raw_jsonl_file", "profile_export_raw.jsonl", id="raw_jsonl_default"),
            param("profile_export_gpu_telemetry_jsonl_file", "gpu_telemetry_export.jsonl", id="gpu_telemetry_jsonl_default"),
        ],
    )  # fmt: skip
    def test_default_export_name(self, attr, expected):
        assert getattr(self._cfg(), attr).name == expected

    def test_server_metrics_defaults(self):
        c = self._cfg()
        assert c.server_metrics_export_jsonl_file.name == "server_metrics_export.jsonl"
        assert c.server_metrics_export_json_file.name == "server_metrics_export.json"
        assert c.server_metrics_export_csv_file.name == "server_metrics_export.csv"
        assert (
            c.server_metrics_export_parquet_file.name == "server_metrics_export.parquet"
        )


class TestPrefixStripsKnownSuffixes:
    """Mirrors main's suffix-stripping: `--profile-export-prefix foo_raw.jsonl`"""

    def test_strips_raw_jsonl_suffix(self):
        cfg = ArtifactsConfig(prefix="foo_raw.jsonl")
        assert cfg.profile_export_jsonl_file.name == "foo.jsonl"
        assert cfg.profile_export_raw_jsonl_file.name == "foo_raw.jsonl"

    def test_strips_timeslices_csv_suffix(self):
        cfg = ArtifactsConfig(prefix="foo_timeslices.csv")
        assert cfg.profile_export_csv_file.name == "foo.csv"
        assert cfg.profile_export_timeslices_csv_file.name == "foo_timeslices.csv"

    def test_strips_server_metrics_parquet_suffix(self):
        cfg = ArtifactsConfig(prefix="foo_server_metrics.parquet")
        assert cfg.profile_export_csv_file.name == "foo.csv"
        assert (
            cfg.server_metrics_export_parquet_file.name == "foo_server_metrics.parquet"
        )


class TestCLIWiringPropagatesPrefix:
    """End-to-end: `--profile-export-prefix foo` on CLI lands as artifacts.prefix='foo'"""

    def test_cli_prefix_applies_to_jsonl_family(self):
        cli = CLIConfig(model_names=["m"], profile_export_prefix="foo")
        cfg = convert_cli_to_aiperf(cli)
        art = cfg.benchmark.artifacts
        assert art.profile_export_jsonl_file.name == "foo.jsonl"
        assert art.profile_export_raw_jsonl_file.name == "foo_raw.jsonl"
        assert (
            art.profile_export_gpu_telemetry_jsonl_file.name
            == "foo_gpu_telemetry.jsonl"
        )
