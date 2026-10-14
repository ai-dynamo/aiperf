# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import orjson
import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import DatasetType
from aiperf.config.dataset import FileDataset
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.config.flags import CLIConfig
from aiperf.config.loader.core import dump_config, load_config_from_string
from aiperf.config.loader.errors import ConfigurationError
from aiperf.config.loader.parsing import parse_file
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli

_PRESIGNED = "https://h.example/t.jsonl?X-Amz-Signature=deadbeefsecret&X-Amz-Expires=60"


def _ds(path: object) -> FileDataset:
    return FileDataset.model_validate(
        {"name": "main", "type": DatasetType.FILE, "path": path}
    )


def test_remote_string_becomes_remote_ref_not_path() -> None:
    ds = _ds("hf://org/repo/t.jsonl")
    assert isinstance(ds.path, RemoteRef)
    assert ds.path.ref == "hf://org/repo/t.jsonl"


def test_local_string_stays_path(tmp_path: Path) -> None:
    assert _ds(str(tmp_path / "x.jsonl")).path == tmp_path / "x.jsonl"


def test_json_round_trip_keeps_reference_and_version() -> None:
    ds = _ds("s3://bucket/k.jsonl")
    ds = ds.model_copy(update={"path": ds.path.with_version('etag:"1"')})
    again = FileDataset.model_validate(orjson.loads(ds.model_dump_json()))
    assert again.path == ds.path


def test_validation_error_hides_presigned_signature() -> None:
    with pytest.raises(ValidationError) as exc_info:
        FileDataset.model_validate(
            {
                "name": "main",
                "type": DatasetType.FILE,
                "path": "ftp://h.example/t.jsonl?X-Amz-Signature=deadbeefsecret",
            }
        )
    assert "deadbeefsecret" not in str(exc_info.value)


def test_validation_error_hides_presigned_signature_in_other_field() -> None:
    with pytest.raises(ValidationError) as exc_info:
        FileDataset.model_validate(
            {
                "name": "main",
                "type": DatasetType.FILE,
                "path": _PRESIGNED,
                "records": [{"text": "x"}],
            }
        )
    assert "deadbeefsecret" not in str(exc_info.value)


def test_yaml_round_trip_keeps_versioned_hf_reference() -> None:
    yaml_text = """\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: file
    format: single_turn
    path: hf://org/repo@v1/t.jsonl
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    cfg = load_config_from_string(yaml_text)
    path = cfg.benchmark.datasets[0].path
    assert path == RemoteRef(ref="hf://org/repo@v1/t.jsonl")

    dumped = dump_config(cfg)
    assert "path: hf://org/repo@v1/t.jsonl" in dumped
    assert load_config_from_string(dumped).benchmark.datasets[0].path == path


def test_parse_file_passes_remote_refs_and_is_idempotent() -> None:
    parsed = parse_file("https://h.example/t.jsonl")
    assert isinstance(parsed, RemoteRef)
    assert parse_file(parsed) is parsed


def test_cli_input_file_remote_reaches_dataset() -> None:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["m"],
            input_file="hf://org/repo/t.jsonl",
            custom_dataset_type=CustomDatasetType.SINGLE_TURN,
        )
    )
    assert run.cfg.get_default_dataset().path == RemoteRef(ref="hf://org/repo/t.jsonl")


def test_remote_dag_without_stop_condition_is_rejected() -> None:
    with pytest.raises((ValueError, ConfigurationError), match="remote reference"):
        make_run_from_cli(
            CLIConfig(
                model_names=["m"],
                input_file="hf://org/repo/dag.jsonl",
                custom_dataset_type=CustomDatasetType.DAG_JSONL,
            )
        )


@pytest.mark.parametrize(
    ("bad_ref", "reason"),
    [
        param(
            "ftp://h.example/t.jsonl?X-Amz-Signature=deadbeefsecret",
            "Unsupported dataset reference scheme",
            id="unsupported-scheme",
        ),
        param(
            "s3://AB/k?X-Amz-Signature=deadbeefsecret",
            "not a valid S3 bucket name",
            id="bad-bucket",
        ),
        param(
            "hf://org/repo?token=deadbeefsecret",
            "cannot carry a query string",
            id="hf-query",
        ),
    ],
)  # fmt: skip
def test_cli_config_error_hides_presigned_secret(bad_ref: str, reason: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        CLIConfig(model_names=["m"], input_file=bad_ref)
    text = str(exc_info.value)
    assert "deadbeefsecret" not in text
    assert reason in text
