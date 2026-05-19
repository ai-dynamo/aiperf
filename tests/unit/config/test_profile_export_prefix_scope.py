# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity regression tests for `--profile-export-prefix`.

On `main`, passing `--profile-export-prefix foo` makes EVERY export file use
`foo` as the base:

    foo.csv                  foo.json                  foo_timeslices.csv
    foo_timeslices.json      foo.jsonl                 foo_raw.jsonl
    foo_gpu_telemetry.jsonl  foo_server_metrics.jsonl  foo_server_metrics.json
    foo_server_metrics.csv   foo_server_metrics.parquet

This branch narrowed the prefix to only the summary/timeslice files. The
JSONL family, GPU telemetry, and server-metrics exports stopped honoring
the prefix. These tests pin the restoration of full-coverage parity.

When `--profile-export-prefix` is NOT given, the historical per-file
defaults remain: `profile_export_aiperf.csv`, `profile_export.jsonl`,
`profile_export_raw.jsonl`, `gpu_telemetry_export.jsonl`, and the
`server_metrics_export.*` family.
"""

from __future__ import annotations

from aiperf.config.artifacts import ArtifactsConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


class TestPrefixAppliedToEveryExport:
    """When prefix is set, EVERY exported file uses `<prefix>...` as its base."""

    def _cfg(self) -> ArtifactsConfig:
        return ArtifactsConfig(prefix="foo")

    def test_csv_summary(self):
        assert self._cfg().profile_export_csv_file.name == "foo.csv"

    def test_json_summary(self):
        assert self._cfg().profile_export_json_file.name == "foo.json"

    def test_timeslices_csv(self):
        assert (
            self._cfg().profile_export_timeslices_csv_file.name == "foo_timeslices.csv"
        )

    def test_timeslices_json(self):
        assert (
            self._cfg().profile_export_timeslices_json_file.name
            == "foo_timeslices.json"
        )

    def test_per_record_jsonl(self):
        assert self._cfg().profile_export_jsonl_file.name == "foo.jsonl"

    def test_raw_jsonl(self):
        assert self._cfg().profile_export_raw_jsonl_file.name == "foo_raw.jsonl"

    def test_gpu_telemetry_jsonl(self):
        assert (
            self._cfg().profile_export_gpu_telemetry_jsonl_file.name
            == "foo_gpu_telemetry.jsonl"
        )

    def test_server_metrics_jsonl(self):
        assert (
            self._cfg().server_metrics_export_jsonl_file.name
            == "foo_server_metrics.jsonl"
        )

    def test_server_metrics_json(self):
        assert (
            self._cfg().server_metrics_export_json_file.name
            == "foo_server_metrics.json"
        )

    def test_server_metrics_csv(self):
        assert (
            self._cfg().server_metrics_export_csv_file.name == "foo_server_metrics.csv"
        )

    def test_server_metrics_parquet(self):
        assert (
            self._cfg().server_metrics_export_parquet_file.name
            == "foo_server_metrics.parquet"
        )


class TestUnsetPrefixPreservesPerFileDefaults:
    """When prefix is not set, historical per-file default names are used."""

    def _cfg(self) -> ArtifactsConfig:
        return ArtifactsConfig()  # no prefix

    def test_csv_summary_default(self):
        assert self._cfg().profile_export_csv_file.name == "profile_export_aiperf.csv"

    def test_json_summary_default(self):
        assert self._cfg().profile_export_json_file.name == "profile_export_aiperf.json"

    def test_timeslices_csv_default(self):
        assert (
            self._cfg().profile_export_timeslices_csv_file.name
            == "profile_export_aiperf_timeslices.csv"
        )

    def test_timeslices_json_default(self):
        assert (
            self._cfg().profile_export_timeslices_json_file.name
            == "profile_export_aiperf_timeslices.json"
        )

    def test_per_record_jsonl_default(self):
        assert self._cfg().profile_export_jsonl_file.name == "profile_export.jsonl"

    def test_raw_jsonl_default(self):
        assert (
            self._cfg().profile_export_raw_jsonl_file.name == "profile_export_raw.jsonl"
        )

    def test_gpu_telemetry_jsonl_default(self):
        assert (
            self._cfg().profile_export_gpu_telemetry_jsonl_file.name
            == "gpu_telemetry_export.jsonl"
        )

    def test_server_metrics_defaults(self):
        c = self._cfg()
        assert c.server_metrics_export_jsonl_file.name == "server_metrics_export.jsonl"
        assert c.server_metrics_export_json_file.name == "server_metrics_export.json"
        assert c.server_metrics_export_csv_file.name == "server_metrics_export.csv"
        assert (
            c.server_metrics_export_parquet_file.name == "server_metrics_export.parquet"
        )


class TestPrefixStripsKnownSuffixes:
    """Mirrors main's suffix-stripping: `--profile-export-prefix foo_raw.jsonl`
    yields a clean `foo` base, not `foo_raw`."""

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
    """End-to-end: `--profile-export-prefix foo` on CLI lands as artifacts.prefix='foo'
    and produces foo-rooted filenames on all exports."""

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
