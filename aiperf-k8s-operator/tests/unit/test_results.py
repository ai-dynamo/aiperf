# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained-run enumeration for the namespace result index."""

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import aiperf_k8s_operator.results as results

ResultsIndex = results.ResultsIndex
ResultIdentity = results.ResultIdentity


async def chunks(body: bytes) -> AsyncIterator[bytes]:
    yield body


def manifest_body(run_id: str, name: str, body: bytes) -> bytes:
    return json.dumps(
        {
            "contractVersion": "native-k8s/v1",
            "runId": run_id,
            "ready": True,
            "wasCancelled": False,
            "artifactRoot": "/results",
            "artifacts": [
                {
                    "path": name,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                    "contentType": "application/octet-stream",
                }
            ],
        },
        separators=(",", ":"),
    ).encode()


async def publish(
    index: ResultsIndex, identity: ResultIdentity, name: str, body: bytes
) -> None:
    assert await index.stage_artifact(
        identity,
        name,
        chunks(body),
        hashlib.sha256(body).hexdigest(),
        len(body),
    )
    document = manifest_body(identity.run_id, name, body)
    assert index.commit_manifest(
        identity,
        document,
        hashlib.sha256(document).hexdigest(),
        len(document),
    )


def test_list_runs_returns_empty_for_unknown_namespace(tmp_path: Path) -> None:
    index = ResultsIndex(tmp_path)
    assert index.list_runs("ns-missing") == []


async def test_list_runs_returns_published_run(tmp_path: Path) -> None:
    index = ResultsIndex(tmp_path)
    identity = ResultIdentity("bench", "job-1", "run-1")
    await publish(index, identity, "answer.bin", b"published")

    listed = index.list_runs("bench")

    assert len(listed) == 1
    entry = listed[0]
    assert entry["jobId"] == "job-1"
    assert entry["runId"] == "run-1"
    assert entry["ready"] is True
    assert entry["artifactCount"] == 1
    assert isinstance(entry["created"], float)


async def test_list_runs_excludes_other_namespaces_newest_first(tmp_path: Path) -> None:
    index = ResultsIndex(tmp_path)
    await publish(index, ResultIdentity("bench", "job-1", "run-1"), "a.bin", b"one")
    await publish(index, ResultIdentity("bench", "job-2", "run-2"), "a.bin", b"two")
    await publish(index, ResultIdentity("other", "job-3", "run-3"), "a.bin", b"three")

    listed = index.list_runs("bench")

    assert {entry["runId"] for entry in listed} == {"run-1", "run-2"}
    assert [entry["created"] for entry in listed] == sorted(
        (entry["created"] for entry in listed), reverse=True
    )
