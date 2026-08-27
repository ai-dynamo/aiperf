# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for MLflow CLI conversion validation.

Ports v1 ``_validate_mlflow_config``:

1. Secondary MLflow flags (--mlflow-experiment / --mlflow-run-name /
   --mlflow-tag / --mlflow-artifact-glob / --mlflow-parent-run-id)
   require --mlflow-tracking-uri to be set.
2. --mlflow-tracking-uri / --mlflow-experiment empty-string rejection.
3. Whitespace normalization on tracking_uri / experiment / run_name /
   artifact_glob entries.
"""

from __future__ import annotations

import pytest

from aiperf.config.flags._converter_telemetry import build_mlflow
from aiperf.config.flags.cli_config import CLIConfig


def _make_cli(**overrides) -> CLIConfig:
    base = {
        "url": "http://localhost:8000/test",
        "model_names": ["test-model"],
    }
    base.update(overrides)
    return CLIConfig(**base)


class TestSecondaryFlagsRequireTrackingUri:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("mlflow_experiment", "my-experiment"),
            ("mlflow_run_name", "my-run"),
            ("mlflow_tags", [("team", "perf")]),
            ("mlflow_parent_run_id", "parent-123"),
            ("mlflow_artifact_globs", ["*.json"]),
        ],
        ids=[
            "experiment",
            "run_name",
            "tags",
            "parent_run_id",
            "artifact_globs",
        ],
    )
    def test_secondary_alone_raises(self, field, value):
        cli = _make_cli(**{field: value})
        with pytest.raises(ValueError, match="require --mlflow-tracking-uri to be set"):
            build_mlflow(cli)

    def test_no_mlflow_flags_returns_empty_dict(self):
        assert build_mlflow(_make_cli()) == {}


class TestEmptyStringRejection:
    def test_empty_tracking_uri_raises(self):
        cli = _make_cli(mlflow_tracking_uri="")
        with pytest.raises(ValueError, match="--mlflow-tracking-uri cannot be empty"):
            build_mlflow(cli)

    def test_whitespace_only_tracking_uri_raises(self):
        cli = _make_cli(mlflow_tracking_uri="   ")
        with pytest.raises(ValueError, match="--mlflow-tracking-uri cannot be empty"):
            build_mlflow(cli)

    def test_empty_experiment_with_tracking_uri_raises(self):
        cli = _make_cli(
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_experiment="   ",
        )
        with pytest.raises(ValueError, match="--mlflow-experiment cannot be empty"):
            build_mlflow(cli)

    def test_artifact_glob_cli_flag_is_singular(self):
        field = CLIConfig.model_fields["mlflow_artifact_globs"]
        param_names = field.metadata[-1].name
        assert param_names == ("--mlflow-artifact-glob",)

    def test_empty_artifact_glob_entry_raises(self):
        cli = _make_cli(
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_artifact_globs=["*.json", "  "],
        )
        with pytest.raises(
            ValueError, match="--mlflow-artifact-glob entries cannot be empty"
        ):
            build_mlflow(cli)


class TestWhitespaceNormalization:
    def test_tracking_uri_stripped(self):
        cli = _make_cli(mlflow_tracking_uri="  http://mlflow:5000  ")
        out = build_mlflow(cli)
        assert out["tracking_uri"] == "http://mlflow:5000"

    def test_experiment_stripped(self):
        cli = _make_cli(
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_experiment="  exp-a  ",
        )
        out = build_mlflow(cli)
        assert out["experiment"] == "exp-a"

    def test_run_name_stripped_or_collapsed_to_none(self):
        cli = _make_cli(
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_run_name="  ",
        )
        out = build_mlflow(cli)
        assert out["run_name"] is None

    def test_run_name_with_content_stripped(self):
        cli = _make_cli(
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_run_name="  baseline  ",
        )
        out = build_mlflow(cli)
        assert out["run_name"] == "baseline"

    def test_artifact_glob_entries_stripped(self):
        cli = _make_cli(
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_artifact_globs=["  *.json  ", "  *.csv"],
        )
        out = build_mlflow(cli)
        assert out["artifact_globs"] == ["*.json", "*.csv"]


class TestSuccessPaths:
    def test_minimal_tracking_uri_only(self):
        cli = _make_cli(mlflow_tracking_uri="http://mlflow:5000")
        out = build_mlflow(cli)
        assert out == {"tracking_uri": "http://mlflow:5000"}

    def test_full_mlflow_config(self):
        cli = _make_cli(
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_experiment="bench",
            mlflow_run_name="run-1",
            mlflow_tags=[("team", "perf")],
            mlflow_parent_run_id="parent-1",
            mlflow_artifact_globs=["*.json"],
        )
        out = build_mlflow(cli)
        assert out["tracking_uri"] == "http://mlflow:5000"
        assert out["experiment"] == "bench"
        assert out["run_name"] == "run-1"
        assert out["tags"] == [("team", "perf")]
        assert out["parent_run_id"] == "parent-1"
        assert out["artifact_globs"] == ["*.json"]
