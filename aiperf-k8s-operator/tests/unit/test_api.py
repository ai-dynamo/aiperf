# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiperf_k8s_operator.api as api
import pytest
from aiperf_k8s_operator.api import create_app
from aiperf_k8s_operator.results import ResultIdentity, ResultsIndex
from aiperf_k8s_operator.settings import OperatorSettings
from httpx import ASGITransport, AsyncClient

IDENTITY = ResultIdentity("bench", "job-1", "run-1")


class StaticLifecycle:
    async def mark_results_ready(
        self, namespace: str, job_id: str, run_id: str
    ) -> None:
        assert (namespace, job_id, run_id) == ("bench", "job-1", "run-1")


class NonResolvingLifecycle:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()

    async def mark_results_ready(
        self, namespace: str, job_id: str, run_id: str
    ) -> None:
        assert (namespace, job_id, run_id) == ("bench", "job-1", "run-1")
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.stopped.set()


class ThreadProbeIndex:
    def stats(self) -> dict[str, int]:
        return {"workerThread": threading.get_ident()}


def upload_headers(body: bytes) -> dict[str, str]:
    digest = hashlib.sha256(body).hexdigest()
    return {
        "X-AIPerf-Content-SHA256": digest,
        "X-AIPerf-Content-Length": str(len(body)),
    }


@asynccontextmanager
async def upload_client(tmp_path: Path) -> AsyncIterator[AsyncClient]:
    settings = OperatorSettings(artifact_root=str(tmp_path))
    async with AsyncClient(
        transport=ASGITransport(
            app=create_app(settings, ResultsIndex(tmp_path), StaticLifecycle())
        ),
        base_url="http://operator.test",
    ) as client:
        yield client


def manifest(artifact: bytes, *, include_missing: bool = False) -> bytes:
    artifacts = [
        {
            "path": "summary.json",
            "sha256": hashlib.sha256(artifact).hexdigest(),
            "bytes": len(artifact),
            "contentType": "application/json",
        }
    ]
    if include_missing:
        artifacts.append(
            {
                "path": "missing.json",
                "sha256": hashlib.sha256(b"missing").hexdigest(),
                "bytes": 7,
                "contentType": "application/json",
            }
        )
    return json.dumps(
        {
            "contractVersion": "native-k8s/v1",
            "runId": "run-1",
            "ready": True,
            "wasCancelled": False,
            "artifactRoot": "/results",
            "artifacts": artifacts,
        },
        separators=(",", ":"),
    ).encode()


async def test_index_stats_are_read_only_and_report_bounded_usage(
    tmp_path: Path,
) -> None:
    async with upload_client(tmp_path) as client:
        response = await client.get("/index/stats")
        assert response.status_code == 200
        assert response.json()["maxPublishedRuns"] > 0
        assert (await client.post("/index/rebuild")).status_code == 404


async def test_index_stats_scans_storage_off_the_event_loop(tmp_path: Path) -> None:
    settings = OperatorSettings(artifact_root=str(tmp_path))
    event_loop_thread = threading.get_ident()
    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings, ThreadProbeIndex())),
        base_url="http://operator.test",
    ) as client:
        response = await client.get("/index/stats")

    assert response.status_code == 200
    assert response.json()["workerThread"] != event_loop_thread


async def test_upload_requires_digest_and_length_but_no_credential(
    tmp_path: Path,
) -> None:
    async with upload_client(tmp_path) as client:
        url = "/api/uploads/bench/job-1/run-1/artifacts/summary.json"
        assert (await client.put(url, content=b"{}")).status_code == 422
        headers = upload_headers(b"{}")
        assert (
            await client.put(url, headers=headers, content=b"tampered")
        ).status_code == 422
        assert (
            await client.put(url, headers=headers, content=b"{}")
        ).status_code == 201


async def test_partial_upload_is_invisible_until_exact_manifest_commit(
    tmp_path: Path,
) -> None:
    async with upload_client(tmp_path) as client:
        artifact = b"{}"
        artifact_url = "/api/uploads/bench/job-1/run-1/artifacts/summary.json"
        assert (
            await client.put(
                artifact_url,
                headers=upload_headers(artifact),
                content=artifact,
            )
        ).status_code == 201
        read_manifest = "/api/results/bench/job-1/run-1/manifest"
        assert (await client.get(read_manifest)).status_code == 409

        incomplete = manifest(artifact, include_missing=True)
        upload_manifest = "/api/uploads/bench/job-1/run-1/manifest"
        assert (
            await client.post(
                upload_manifest,
                headers=upload_headers(incomplete),
                content=incomplete,
            )
        ).status_code == 422

        complete = manifest(artifact)
        assert (
            await client.post(
                upload_manifest,
                headers=upload_headers(complete),
                content=complete,
            )
        ).status_code == 201
        assert (await client.get(read_manifest)).json()["runId"] == "run-1"


async def test_durable_manifest_ack_does_not_wait_for_lifecycle_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "_LIFECYCLE_TIMEOUT_SECONDS", 0.01)
    lifecycle = NonResolvingLifecycle()
    settings = OperatorSettings(artifact_root=str(tmp_path))
    async with AsyncClient(
        transport=ASGITransport(
            app=create_app(settings, ResultsIndex(tmp_path), lifecycle)
        ),
        base_url="http://operator.test",
    ) as client:
        artifact = b"{}"
        assert (
            await client.put(
                "/api/uploads/bench/job-1/run-1/artifacts/summary.json",
                headers=upload_headers(artifact),
                content=artifact,
            )
        ).status_code == 201
        document = manifest(artifact)
        response = await asyncio.wait_for(
            client.post(
                "/api/uploads/bench/job-1/run-1/manifest",
                headers=upload_headers(document),
                content=document,
            ),
            timeout=0.25,
        )

        assert response.status_code == 201
        await asyncio.wait_for(lifecycle.started.wait(), timeout=0.25)
        await asyncio.wait_for(lifecycle.stopped.wait(), timeout=0.25)


async def test_identical_replay_is_idempotent_but_corruption_is_not_acked(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    settings = OperatorSettings(artifact_root=str(tmp_path))
    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings, index, StaticLifecycle())),
        base_url="http://operator.test",
    ) as client:
        artifact = b"{}"
        artifact_url = "/api/uploads/bench/job-1/run-1/artifacts/summary.json"
        artifact_headers = upload_headers(artifact)
        assert (
            await client.put(artifact_url, headers=artifact_headers, content=artifact)
        ).status_code == 201
        assert (
            await client.put(artifact_url, headers=artifact_headers, content=artifact)
        ).status_code == 200
        document = manifest(artifact)
        document_headers = upload_headers(document)
        url = "/api/uploads/bench/job-1/run-1/manifest"
        assert (
            await client.post(url, headers=document_headers, content=document)
        ).status_code == 201
        assert (
            await client.post(url, headers=document_headers, content=document)
        ).status_code == 200

        handle = index.open_artifact(IDENTITY, "summary.json")
        descriptor_path = Path(f"/proc/self/fd/{handle.file.fileno()}").resolve()
        handle.close()
        descriptor_path.write_bytes(b"corrupt")
        assert (
            await client.post(url, headers=document_headers, content=document)
        ).status_code == 422
