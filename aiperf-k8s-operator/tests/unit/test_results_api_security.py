# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import aiperf_k8s_operator.api as api_module
from aiperf_k8s_operator.results import ResultIdentity, ResultsIndex, StorageLimits
from aiperf_k8s_operator.settings import OperatorSettings
from httpx import ASGITransport, AsyncClient

IDENTITY = ResultIdentity("bench", "job-1", "run-1")


class StaticLifecycle:
    def __init__(self) -> None:
        self.ready: list[tuple[str, str, str]] = []
        self.completed = asyncio.Event()

    async def mark_results_ready(
        self, namespace: str, job_id: str, run_id: str
    ) -> None:
        self.ready.append((namespace, job_id, run_id))
        self.completed.set()


async def chunks(body: bytes) -> AsyncIterator[bytes]:
    yield body


def manifest(body: bytes) -> bytes:
    return json.dumps(
        {
            "contractVersion": "native-k8s/v1",
            "runId": "run-1",
            "ready": True,
            "wasCancelled": False,
            "artifactRoot": "/results",
            "artifacts": [
                {
                    "path": "summary.bin",
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                    "contentType": "application/octet-stream",
                }
            ],
        },
        separators=(",", ":"),
    ).encode()


async def publish(index: ResultsIndex, body: bytes) -> None:
    assert await index.stage_artifact(
        IDENTITY,
        "summary.bin",
        chunks(body),
        hashlib.sha256(body).hexdigest(),
        len(body),
    )
    document = manifest(body)
    assert index.commit_manifest(
        IDENTITY,
        document,
        hashlib.sha256(document).hexdigest(),
        len(document),
    )


def upload_headers(body: bytes) -> dict[str, str]:
    digest = hashlib.sha256(body).hexdigest()
    return {
        "X-AIPerf-Content-SHA256": digest,
        "X-AIPerf-Content-Length": str(len(body)),
    }


def app(tmp_path: Path, index: ResultsIndex, lifecycle: StaticLifecycle):
    return api_module.create_app(
        OperatorSettings(artifact_root=str(tmp_path)),
        index,
        lifecycle,
    )


async def test_direct_result_reads_use_exact_triple_without_application_auth(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    await publish(index, b"result")
    async with AsyncClient(
        transport=ASGITransport(app=app(tmp_path, index, StaticLifecycle())),
        base_url="http://operator.test",
    ) as client:
        path = "/api/results/bench/job-1/run-1/manifest"
        response = await client.get(path)
        assert response.status_code == 200
        assert response.json()["runId"] == "run-1"
        assert (
            await client.get("/api/results/other/job-1/run-1/manifest")
        ).status_code == 409
        assert (await client.get("/runs/run-1/manifest")).status_code == 404


async def test_artifact_open_and_hash_run_off_loop_and_response_streams(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    body = b"x" * (1024 * 1024 + 7)
    await publish(index, body)
    event_loop_thread = threading.get_ident()
    open_threads: list[int] = []
    original = index.open_artifact

    def observed_open(identity: ResultIdentity, name: str):
        open_threads.append(threading.get_ident())
        return original(identity, name)

    index.open_artifact = observed_open  # type: ignore[method-assign]
    async with AsyncClient(
        transport=ASGITransport(app=app(tmp_path, index, StaticLifecycle())),
        base_url="http://operator.test",
    ) as client:
        response = await client.get(
            "/api/results/bench/job-1/run-1/artifacts/summary.bin"
        )
    assert response.status_code == 200
    assert response.content == body
    assert open_threads and open_threads[0] != event_loop_thread


async def test_manifest_commit_runs_off_loop_and_marks_lifecycle_ready(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    lifecycle = StaticLifecycle()
    commit_threads: list[int] = []
    event_loop_thread = threading.get_ident()
    original = index.commit_manifest

    def observed_commit(*args, **kwargs):
        commit_threads.append(threading.get_ident())
        return original(*args, **kwargs)

    index.commit_manifest = observed_commit  # type: ignore[method-assign]
    async with AsyncClient(
        transport=ASGITransport(app=app(tmp_path, index, lifecycle)),
        base_url="http://operator.test",
    ) as client:
        artifact = b"result"
        artifact_url = "/api/uploads/bench/job-1/run-1/artifacts/summary.bin"
        assert (
            await client.put(
                artifact_url,
                headers=upload_headers(artifact),
                content=artifact,
            )
        ).status_code == 201
        document = manifest(artifact)
        assert (
            await client.post(
                "/api/uploads/bench/job-1/run-1/manifest",
                headers=upload_headers(document),
                content=document,
            )
        ).status_code == 201
        await asyncio.wait_for(lifecycle.completed.wait(), timeout=0.25)
    assert commit_threads and commit_threads[0] != event_loop_thread
    assert lifecycle.ready == [("bench", "job-1", "run-1")]


async def test_expired_completed_results_return_gone(
    tmp_path: Path,
) -> None:
    now = [100.0]
    index = ResultsIndex(
        tmp_path,
        limits=StorageLimits(published_ttl_seconds=10),
        now=lambda: now[0],
    )
    await publish(index, b"result")
    now[0] = 111.0

    async with AsyncClient(
        transport=ASGITransport(app=app(tmp_path, index, StaticLifecycle())),
        base_url="http://operator.test",
    ) as client:
        response = await client.get("/api/results/bench/job-1/run-1/manifest")

    assert response.status_code == 410
