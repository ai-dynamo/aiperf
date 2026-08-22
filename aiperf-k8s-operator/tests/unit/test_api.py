# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from aiperf_k8s_operator.api import create_app
from aiperf_k8s_operator.results import ResultsIndex
from aiperf_k8s_operator.settings import OperatorSettings


def client(tmp_path: Path) -> TestClient:
    settings = OperatorSettings(artifact_root=str(tmp_path), index_rebuild_token="rebuild-token")
    return TestClient(create_app(settings, ResultsIndex(tmp_path)))


def test_unready_manifest_is_conflict(tmp_path: Path) -> None:
    assert client(tmp_path).get("/runs/run-1/manifest").status_code == 409


def test_undeclared_artifact_is_not_found(tmp_path: Path) -> None:
    index = ResultsIndex(tmp_path)
    index.publish_manifest("run-1", {"artifacts": []})
    app = create_app(OperatorSettings(artifact_root=str(tmp_path), index_rebuild_token="token"), index)
    assert TestClient(app).get("/runs/run-1/artifacts/private.json").status_code == 404


def test_declared_artifact_and_rebuild_authentication(tmp_path: Path) -> None:
    artifact = tmp_path / "run-1" / "summary.json"
    artifact.parent.mkdir()
    artifact.write_bytes(b"{}")
    index = ResultsIndex(tmp_path)
    index.publish_manifest("run-1", {"artifacts": [{"path": "summary.json", "sha256": hashlib.sha256(b"{}").hexdigest(), "contentType": "application/json"}]})
    api = TestClient(create_app(OperatorSettings(artifact_root=str(tmp_path), index_rebuild_token="token"), index))
    assert api.get("/runs/run-1/artifacts/summary.json").content == b"{}"
    assert api.post("/index/rebuild").status_code == 401
    assert api.post("/index/rebuild", headers={"Authorization": "Bearer token"}).status_code == 200
