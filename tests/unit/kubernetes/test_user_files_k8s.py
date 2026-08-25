# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""artifacts.userFiles on the Kubernetes path: declared files must materialize.

Regression cover for the silent no-op where an AIPerfJob declaring
``artifacts.userFiles`` produced neither files nor an error, because
``aiperf service`` boots from the serialized run and skips the resolver chain
that writes them locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.user_files import RunMeta, UserFileError
from aiperf.kubernetes.resources import ConfigMapSpec
from aiperf.kubernetes.serialized_run import read_serialized_run_json
from aiperf.kubernetes.spec_converter import AIPerfJobSpecConverter, build_benchmark_run
from aiperf.kubernetes.user_files import (
    materialize_serialized_run_user_files,
    resolve_run_meta,
)

_USER_FILES = [
    {
        "path": "input_config.json",
        "format": "json",
        "content": {"model": "{{ model }}", "endpoint": "{{ endpoint_url }}"},
    },
    {
        "path": "meta/notes.md",
        "format": "text",
        "content": "Run {{ job_name }} in {{ namespace }} at {{ epoch }}.\n",
    },
]


def _job_spec(user_files: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Build a minimal AIPerfJob spec, optionally carrying artifacts.userFiles."""
    artifacts: dict[str, Any] = {}
    if user_files is not None:
        artifacts["userFiles"] = user_files
    return {
        "image": "nvcr.io/nvidia/aiperf:latest",
        "benchmark": {
            "models": ["my-org/my-model"],
            "endpoint": {"type": "chat", "urls": ["http://frontend:8000"]},
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 2,
                    "requests": 4,
                }
            ],
            "artifacts": artifacts,
        },
    }


def _serialized_pod_run(
    spec: dict[str, Any],
    *,
    run_meta: RunMeta | None,
) -> BenchmarkRun:
    """Round-trip a spec through the real controller-side ConfigMap rendering.

    Mirrors ``operator/handlers/create.py`` (build_benchmark_run ->
    ConfigMapSpec.from_benchmark_run) and then ``cli_commands/service.py``
    (read_serialized_run_json -> BenchmarkRun.model_validate), so the test
    exercises the same serialization boundary a real pod does.
    """
    converter = AIPerfJobSpecConverter(
        spec, "bench-job", "bench-ns", job_id="bench-job"
    )
    config = converter.to_aiperf_config()
    run = build_benchmark_run(
        run_config=config.model_dump(
            mode="python",
            exclude_unset=True,
            exclude_none=True,
            context={"include_secrets": True},
        ),
        run_id="bench-job",
        namespace="bench-ns",
        run_meta=run_meta,
    )
    configmap = ConfigMapSpec.from_benchmark_run(
        name="aiperf-bench-job-config",
        namespace="bench-ns",
        run=run,
        job_id="bench-job",
    )
    return BenchmarkRun.model_validate(orjson.loads(configmap.data["run_config.json"]))


def _repoint(run: BenchmarkRun, run_dir: Path) -> BenchmarkRun:
    """Move the run off the ``/results`` container mount onto a writable path."""
    run.artifact_dir = run_dir
    run.cfg.artifacts.dir = run_dir
    return run


def test_materialize_serialized_run_user_files_declared_files_writes_them(
    tmp_path: Path,
) -> None:
    run = _serialized_pod_run(
        _job_spec(_USER_FILES),
        run_meta=RunMeta(
            epoch="1714069323", job_name="bench-job", namespace="bench-ns"
        ),
    )
    run_dir = tmp_path / "results"

    materialize_serialized_run_user_files(_repoint(run, run_dir))

    assert orjson.loads((run_dir / "input_config.json").read_bytes()) == {
        "model": "my-org/my-model",
        "endpoint": "http://frontend:8000",
    }
    assert (run_dir / "meta" / "notes.md").read_text() == (
        "Run bench-job in bench-ns at 1714069323.\n"
    )


def test_materialize_serialized_run_user_files_no_declaration_is_noop(
    tmp_path: Path,
) -> None:
    run = _serialized_pod_run(_job_spec(None), run_meta=None)
    run_dir = tmp_path / "results"

    materialize_serialized_run_user_files(_repoint(run, run_dir))

    assert not run_dir.exists()


def test_materialize_serialized_run_user_files_undefined_variable_raises(
    tmp_path: Path,
) -> None:
    """A bad template must fail loudly before the benchmark starts, never silently."""
    run = _serialized_pod_run(
        _job_spec([{"path": "bad.md", "format": "text", "content": "{{ nope }}"}]),
        run_meta=RunMeta(
            epoch="1714069323", job_name="bench-job", namespace="bench-ns"
        ),
    )

    with pytest.raises(UserFileError, match="bad.md"):
        materialize_serialized_run_user_files(_repoint(run, tmp_path / "results"))


def test_build_benchmark_run_run_meta_survives_configmap_serialization() -> None:
    """RunMeta must reach the pod: /results carries no epoch or job name."""
    run = _serialized_pod_run(
        _job_spec(_USER_FILES),
        run_meta=RunMeta(
            epoch="1714069323", job_name="bench-job", namespace="bench-ns"
        ),
    )

    assert run.run_meta == RunMeta(
        epoch="1714069323", job_name="bench-job", namespace="bench-ns"
    )


def test_resolve_run_meta_serialized_meta_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIPERF_JOB_ID", "env-job")
    monkeypatch.setenv("AIPERF_NAMESPACE", "env-ns")
    run = _serialized_pod_run(
        _job_spec(_USER_FILES),
        run_meta=RunMeta(epoch="1714069323", job_name="cr-job", namespace="cr-ns"),
    )

    assert resolve_run_meta(run).job_name == "cr-job"
    assert resolve_run_meta(run).namespace == "cr-ns"


@pytest.mark.parametrize(
    "job_id_env,expected_job_name",
    [
        param("direct-job", "direct-job", id="job_id_env_set"),
        param(None, "bench-job", id="job_id_env_absent_falls_back_to_benchmark_id"),
    ],
)  # fmt: skip
def test_resolve_run_meta_without_serialized_meta_uses_pod_identity(
    monkeypatch: pytest.MonkeyPatch,
    job_id_env: str | None,
    expected_job_name: str,
) -> None:
    """Direct mode (--no-operator) has no CR, so identity comes from pod env."""
    monkeypatch.delenv("AIPERF_JOB_ID", raising=False)
    if job_id_env is not None:
        monkeypatch.setenv("AIPERF_JOB_ID", job_id_env)
    monkeypatch.setenv("AIPERF_NAMESPACE", "direct-ns")
    run = _serialized_pod_run(_job_spec(_USER_FILES), run_meta=None)

    meta = resolve_run_meta(run)

    assert run.run_meta is None
    assert meta.job_name == expected_job_name
    assert meta.namespace == "direct-ns"
    assert meta.epoch.isdigit()


def test_read_serialized_run_json_pod_path_carries_user_files(tmp_path: Path) -> None:
    """The declared files travel inside run_config.json, not via re-resolution."""
    run = _serialized_pod_run(
        _job_spec(_USER_FILES),
        run_meta=RunMeta(
            epoch="1714069323", job_name="bench-job", namespace="bench-ns"
        ),
    )
    path = tmp_path / "run_config.json"
    path.write_bytes(orjson.dumps(run.model_dump(mode="json", exclude_none=True)))

    reloaded = BenchmarkRun.model_validate(
        orjson.loads(read_serialized_run_json(path) or "{}")
    )

    assert [f.path for f in reloaded.cfg.artifacts.user_files] == [
        "input_config.json",
        "meta/notes.md",
    ]
