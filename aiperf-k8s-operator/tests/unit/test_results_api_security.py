# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import aiperf_k8s_operator.api as api_module
from aiperf_k8s_operator.results import ResultIdentity, ResultsIndex, StorageLimits
from aiperf_k8s_operator.settings import OperatorSettings
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from httpx import ASGITransport, AsyncClient

OBJECT_UID = "9d2f3e2a-1111-4222-8333-abcdefabcdef"
UPLOAD_PRIVATE_SEED = bytes.fromhex(
    "cb0c5712ee5b05b22c22b136db935113f7b7d7a29356737e7f030be19cfabbf6"
)
UPLOAD_PUBLIC_KEY = "8uFXJpCIj094psVHjvxpu5_YA6Ruivm9sb8z4GNRlTo"
READ_RAW = bytes(range(32))
READ_TOKEN = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
READ_TOKEN_SHA256 = "630dcd2966c4336691125448bbb25b4ff412a49c732db2c8abc1b8581bd710dd"
IDENTITY = ResultIdentity("bench", "job-1", "run-1", OBJECT_UID)


class StaticAuthorities:
    def __init__(self) -> None:
        self.ready: list[tuple[str, str, str, str]] = []

    async def authorities(self, namespace: str, job_id: str, run_id: str):
        authority_type = getattr(api_module, "RunAuthorities", None)
        assert authority_type is not None
        if (namespace, job_id, run_id) != (
            IDENTITY.namespace,
            IDENTITY.job_id,
            IDENTITY.run_id,
        ):
            return None
        return authority_type(OBJECT_UID, UPLOAD_PUBLIC_KEY, READ_TOKEN_SHA256)

    async def mark_results_ready(
        self, namespace: str, job_id: str, run_id: str, object_uid: str
    ) -> None:
        self.ready.append((namespace, job_id, run_id, object_uid))


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


def upload_headers(kind: str, path: str, body: bytes) -> dict[str, str]:
    digest = hashlib.sha256(body).hexdigest()
    fields = (
        "bench",
        "job-1",
        "run-1",
        OBJECT_UID,
        kind,
        path,
        digest,
        str(len(body)),
    )
    message = bytearray(b"AIPERF-RESULTS-UPLOAD-SIGNATURE\x01")
    for field in fields:
        encoded = field.encode()
        message.extend(len(encoded).to_bytes(8, "big"))
        message.extend(encoded)
    signature = Ed25519PrivateKey.from_private_bytes(UPLOAD_PRIVATE_SEED).sign(message)
    return {
        "X-AIPerf-Content-SHA256": digest,
        "X-AIPerf-Content-Length": str(len(body)),
        "X-AIPerf-Signature": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode(),
    }


def app(tmp_path: Path, index: ResultsIndex, authorities: StaticAuthorities):
    return api_module.create_app(
        OperatorSettings(artifact_root=str(tmp_path), index_rebuild_token="admin"),
        index,
        authorities,
    )


async def test_direct_result_reads_require_exact_triple_and_capability(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    await publish(index, b"result")
    async with AsyncClient(
        transport=ASGITransport(app=app(tmp_path, index, StaticAuthorities())),
        base_url="http://operator.test",
    ) as client:
        path = "/api/results/bench/job-1/run-1/manifest"
        assert (await client.get(path)).status_code == 401
        assert (
            await client.get(path, headers={"Authorization": "Bearer wrong"})
        ).status_code == 401
        response = await client.get(
            path, headers={"Authorization": f"Bearer {READ_TOKEN}"}
        )
        assert response.status_code == 200
        assert response.json()["runId"] == "run-1"
        assert (
            await client.get(
                path, headers={"X-AIPerf-Results-Token": READ_TOKEN}
            )
        ).status_code == 200
        assert (
            await client.get(
                path,
                headers={
                    "Authorization": f"Bearer {READ_TOKEN}",
                    "X-AIPerf-Results-Token": READ_TOKEN,
                },
            )
        ).status_code == 401
        assert (
            await client.get(
                "/api/results/other/job-1/run-1/manifest",
                headers={"Authorization": f"Bearer {READ_TOKEN}"},
            )
        ).status_code == 401
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
        transport=ASGITransport(app=app(tmp_path, index, StaticAuthorities())),
        base_url="http://operator.test",
    ) as client:
        response = await client.get(
            "/api/results/bench/job-1/run-1/artifacts/summary.bin",
            headers={"Authorization": f"Bearer {READ_TOKEN}"},
        )
    assert response.status_code == 200
    assert response.content == body
    assert open_threads and open_threads[0] != event_loop_thread


async def test_upload_authority_and_commit_are_bound_to_current_object_uid(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    authorities = StaticAuthorities()
    commit_threads: list[int] = []
    event_loop_thread = threading.get_ident()
    original = index.commit_manifest

    def observed_commit(*args, **kwargs):
        commit_threads.append(threading.get_ident())
        return original(*args, **kwargs)

    index.commit_manifest = observed_commit  # type: ignore[method-assign]
    async with AsyncClient(
        transport=ASGITransport(app=app(tmp_path, index, authorities)),
        base_url="http://operator.test",
    ) as client:
        artifact = b"result"
        artifact_url = "/api/uploads/bench/job-1/run-1/artifacts/summary.bin"
        assert (
            await client.put(
                artifact_url,
                headers=upload_headers("artifact", "summary.bin", artifact),
                content=artifact,
            )
        ).status_code == 201
        document = manifest(artifact)
        assert (
            await client.post(
                "/api/uploads/bench/job-1/run-1/manifest",
                headers=upload_headers("manifest", "results-manifest.json", document),
                content=document,
            )
        ).status_code == 201
    assert commit_threads and commit_threads[0] != event_loop_thread
    assert authorities.ready == [("bench", "job-1", "run-1", OBJECT_UID)]


async def test_expired_completed_results_return_gone_while_authority_exists(
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
        transport=ASGITransport(app=app(tmp_path, index, StaticAuthorities())),
        base_url="http://operator.test",
    ) as client:
        response = await client.get(
            "/api/results/bench/job-1/run-1/manifest",
            headers={"Authorization": f"Bearer {READ_TOKEN}"},
        )

    assert response.status_code == 410
