# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import orjson
import pytest

from aiperf.common.redact import REDACTED_VALUE
from aiperf.config.artifacts import OutputDefaults
from aiperf.config.config import BenchmarkConfig
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.config.flags import CLIConfig
from aiperf.dataset.provenance import remote_dataset_provenance
from aiperf.dataset.remote.index import write_index
from aiperf.dataset.remote.models import ProbeResult, RemoteFile, ResolvedSource
from aiperf.exporters.metrics_json_exporter import MetricsJsonExporter
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli
from tests.unit.exporters.conftest import make_exporter_config

URL = "https://h.example/t.jsonl?X-Amz-Signature=secret"


def _cfg(ref: RemoteRef) -> BenchmarkConfig:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["m"],
            input_file=URL,
            custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            disable_auto_fixed_schedule=True,
        )
    )
    run.cfg.datasets[0] = run.cfg.datasets[0].model_copy(update={"path": ref})
    return run.cfg


def _materialize(tmp_path: Path, ref: RemoteRef) -> None:
    local = tmp_path / "t.jsonl"
    local.write_bytes(b"{}\n")
    probe = ProbeResult(
        ref=ref, pinned_ref=ref.display, data_relpath=None, data_is_dir=False,
        filename="t.jsonl", manifest=None, manifest_sha256="m" * 64, needs_auth=False,
    )  # fmt: skip
    write_index(
        ResolvedSource(
            probe=probe,
            local_path=local,
            files=[RemoteFile(path="t.jsonl", sha256="f" * 64, size_bytes=3)],
        )
    )


def test_materialized_remote_provenance(tmp_path: Path) -> None:
    ref = RemoteRef(ref=URL).with_version("etag:1")
    _materialize(tmp_path, ref)
    prov = remote_dataset_provenance(_cfg(ref))
    assert prov == {
        "source_type": "remote_file",
        "ref": ref.display,
        "pinned_ref": ref.display,
        "resolved_version": "etag:1",
        "format": "mooncake_trace",
        "files": [{"path": "t.jsonl", "sha256": "f" * 64, "size_bytes": 3}],
        "manifest_sha256": "m" * 64,
    }


def test_presigned_provenance_has_no_signature(tmp_path: Path) -> None:
    ref = RemoteRef(ref=URL).with_version("etag:1")
    _materialize(tmp_path, ref)
    for prov in (remote_dataset_provenance(_cfg(ref)), _unmaterialized_presigned()):
        assert prov is not None
        assert b"secret" not in orjson.dumps(prov)
        assert REDACTED_VALUE in str(prov["ref"])


def _unmaterialized_presigned() -> dict[str, object] | None:
    ref = RemoteRef(ref=URL).with_version("never-materialized")
    return remote_dataset_provenance(_cfg(ref))


def test_unmaterialized_remote_returns_minimal_record() -> None:
    prov = remote_dataset_provenance(
        _cfg(RemoteRef(ref="hf://org/repo/t.jsonl").with_version("abc"))
    )
    assert prov == {
        "source_type": "remote_file",
        "ref": "hf://org/repo/t.jsonl",
        "resolved_version": "abc",
    }


def test_unpinned_remote_returns_minimal_record() -> None:
    prov = remote_dataset_provenance(_cfg(RemoteRef(ref="hf://org/repo/t.jsonl")))
    assert prov == {
        "source_type": "remote_file",
        "ref": "hf://org/repo/t.jsonl",
        "resolved_version": None,
    }


def test_local_dataset_has_no_remote_provenance(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    f.write_text("{}\n")
    run = make_run_from_cli(CLIConfig(model_names=["m"], input_file=str(f)))
    assert remote_dataset_provenance(run.cfg) is None


@pytest.mark.asyncio
async def test_json_export_records_remote_provenance_without_signature(
    tmp_path: Path,
) -> None:
    ref = RemoteRef(ref=URL).with_version("etag:1")
    _materialize(tmp_path, ref)

    class _Results:
        metrics: list = []
        records: list = []
        start_ns = None
        end_ns = None
        has_results = False
        was_cancelled = False
        is_complete = True
        incomplete_reason = None
        error_summary: list = []

    out = tmp_path / "out"
    out.mkdir()
    exporter = MetricsJsonExporter(
        make_exporter_config(
            results=_Results(),
            cfg=_cfg(ref),
            artifact_directory=out,
            telemetry_results=None,
        )
    )
    await exporter.export()

    text = (out / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE).read_text()
    assert "secret" not in text
    dataset = orjson.loads(text)["metadata"]["dataset"]
    assert dataset["source_type"] == "remote_file"
    assert dataset["files"] == [
        {"path": "t.jsonl", "sha256": "f" * 64, "size_bytes": 3}
    ]


def test_hf_parquet_layout_is_recorded_in_provenance(tmp_path: Path) -> None:
    ref = RemoteRef(ref=URL).with_version("etag:1")
    shard = tmp_path / "d" / "train-00000-of-00001.parquet"
    shard.parent.mkdir()
    shard.write_bytes(b"x")
    probe = ProbeResult(
        ref=ref, pinned_ref=ref.display, data_relpath="d", data_is_dir=True,
        filename="d", manifest=None, manifest_sha256=None, needs_auth=False,
        data_files=["d/train-00000-of-00001.parquet"],
    )  # fmt: skip
    write_index(
        ResolvedSource(
            probe=probe,
            local_path=shard.parent,
            files=[
                RemoteFile(
                    path="d/train-00000-of-00001.parquet", sha256="f" * 64, size_bytes=1
                )
            ],
        )
    )
    assert remote_dataset_provenance(_cfg(ref))["layout"] == "hf_parquet"
