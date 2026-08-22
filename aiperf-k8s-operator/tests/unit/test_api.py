# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from aiperf_k8s_operator.api import RunAuthorities, create_app
from aiperf_k8s_operator.results import ResultIdentity, ResultsIndex
from aiperf_k8s_operator.settings import OperatorSettings
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from httpx import ASGITransport, AsyncClient

OBJECT_UID = "9d2f3e2a-1111-4222-8333-abcdefabcdef"
PRIVATE_SEED = bytes.fromhex(
    "cb0c5712ee5b05b22c22b136db935113f7b7d7a29356737e7f030be19cfabbf6"
)
PUBLIC_KEY = "8uFXJpCIj094psVHjvxpu5_YA6Ruivm9sb8z4GNRlTo"
READ_TOKEN = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
READ_DIGEST = "630dcd2966c4336691125448bbb25b4ff412a49c732db2c8abc1b8581bd710dd"
IDENTITY = ResultIdentity("bench", "job-1", "run-1", OBJECT_UID)


class StaticAuthorities:
    async def authorities(
        self, namespace: str, job_id: str, run_id: str
    ) -> RunAuthorities | None:
        if (namespace, job_id, run_id) == ("bench", "job-1", "run-1"):
            return RunAuthorities(OBJECT_UID, PUBLIC_KEY, READ_DIGEST)
        return None

    async def mark_results_ready(
        self, namespace: str, job_id: str, run_id: str, object_uid: str
    ) -> None:
        assert (namespace, job_id, run_id, object_uid) == (
            "bench",
            "job-1",
            "run-1",
            OBJECT_UID,
        )


def _framed(domain: bytes, *fields: str) -> bytes:
    message = bytearray(domain)
    for field in fields:
        value = field.encode()
        message.extend(len(value).to_bytes(8, "big"))
        message.extend(value)
    return bytes(message)


def upload_headers(kind: str, path: str, body: bytes) -> dict[str, str]:
    digest = hashlib.sha256(body).hexdigest()
    message = _framed(
        b"AIPERF-RESULTS-UPLOAD-SIGNATURE\x01",
        "bench",
        "job-1",
        "run-1",
        OBJECT_UID,
        kind,
        path,
        digest,
        str(len(body)),
    )
    signature = Ed25519PrivateKey.from_private_bytes(PRIVATE_SEED).sign(message)
    return {
        "X-AIPerf-Content-SHA256": digest,
        "X-AIPerf-Content-Length": str(len(body)),
        "X-AIPerf-Signature": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode(),
    }


@asynccontextmanager
async def upload_client(tmp_path: Path) -> AsyncIterator[AsyncClient]:
    settings = OperatorSettings(
        artifact_root=str(tmp_path), index_rebuild_token="rebuild-token"
    )
    async with AsyncClient(
        transport=ASGITransport(
            app=create_app(settings, ResultsIndex(tmp_path), StaticAuthorities())
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


async def test_admin_index_routes_require_the_separate_admin_token(
    tmp_path: Path,
) -> None:
    async with upload_client(tmp_path) as client:
        assert (await client.get("/index/stats")).status_code == 401
        assert (
            await client.get(
                "/index/stats", headers={"Authorization": "Bearer rebuild-token"}
            )
        ).status_code == 200
        assert (await client.post("/index/rebuild")).status_code == 401


async def test_upload_requires_authority_bound_to_exact_path_and_body(
    tmp_path: Path,
) -> None:
    async with upload_client(tmp_path) as client:
        url = "/api/uploads/bench/job-1/run-1/artifacts/summary.json"
        assert (await client.put(url, content=b"{}")).status_code == 401
        headers = upload_headers("artifact", "summary.json", b"{}")
        assert (
            await client.put(url, headers=headers, content=b"tampered")
        ).status_code == 422
        assert (
            await client.put(
                "/api/uploads/bench/job-1/run-1/artifacts/other.json",
                headers=headers,
                content=b"{}",
            )
        ).status_code == 401
        assert (await client.put(url, headers=headers, content=b"{}")).status_code == 201


async def test_partial_upload_is_invisible_until_exact_manifest_commit(
    tmp_path: Path,
) -> None:
    async with upload_client(tmp_path) as client:
        artifact = b"{}"
        artifact_url = "/api/uploads/bench/job-1/run-1/artifacts/summary.json"
        assert (
            await client.put(
                artifact_url,
                headers=upload_headers("artifact", "summary.json", artifact),
                content=artifact,
            )
        ).status_code == 201
        read_headers = {"Authorization": f"Bearer {READ_TOKEN}"}
        read_manifest = "/api/results/bench/job-1/run-1/manifest"
        assert (await client.get(read_manifest, headers=read_headers)).status_code == 409

        incomplete = manifest(artifact, include_missing=True)
        upload_manifest = "/api/uploads/bench/job-1/run-1/manifest"
        assert (
            await client.post(
                upload_manifest,
                headers=upload_headers(
                    "manifest", "results-manifest.json", incomplete
                ),
                content=incomplete,
            )
        ).status_code == 422

        complete = manifest(artifact)
        assert (
            await client.post(
                upload_manifest,
                headers=upload_headers("manifest", "results-manifest.json", complete),
                content=complete,
            )
        ).status_code == 201
        assert (
            await client.get(read_manifest, headers=read_headers)
        ).json()["runId"] == "run-1"


async def test_identical_replay_is_idempotent_but_corruption_is_not_acked(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    settings = OperatorSettings(
        artifact_root=str(tmp_path), index_rebuild_token="rebuild-token"
    )
    async with AsyncClient(
        transport=ASGITransport(
            app=create_app(settings, index, StaticAuthorities())
        ),
        base_url="http://operator.test",
    ) as client:
        artifact = b"{}"
        artifact_url = "/api/uploads/bench/job-1/run-1/artifacts/summary.json"
        artifact_headers = upload_headers("artifact", "summary.json", artifact)
        assert (
            await client.put(artifact_url, headers=artifact_headers, content=artifact)
        ).status_code == 201
        assert (
            await client.put(artifact_url, headers=artifact_headers, content=artifact)
        ).status_code == 200
        document = manifest(artifact)
        document_headers = upload_headers(
            "manifest", "results-manifest.json", document
        )
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
