# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

import aiperf_k8s_operator.results as results
import pytest

ResultsIndex = results.ResultsIndex
UploadConflict = results.UploadConflict
UploadTooLarge = results.UploadTooLarge


async def chunks(body: bytes) -> AsyncIterator[bytes]:
    yield body


def manifest(run_id: str, name: str, body: bytes) -> bytes:
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
    index: ResultsIndex, identity: object, name: str, body: bytes
) -> None:
    assert await index.stage_artifact(
        identity,
        name,
        chunks(body),
        hashlib.sha256(body).hexdigest(),
        len(body),
    )
    document = manifest(identity.run_id, name, body)
    assert index.commit_manifest(
        identity,
        document,
        hashlib.sha256(document).hexdigest(),
        len(document),
    )


async def test_published_storage_identity_is_namespace_job_and_run(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    identity_type = getattr(results, "ResultIdentity", None)
    assert identity_type is not None
    first = identity_type("team-a", "job-a", "shared-run")
    second = identity_type("team-b", "job-b", "shared-run")
    await publish(index, first, "answer.bin", b"first")
    await publish(index, second, "answer.bin", b"second")

    assert index.ready_manifest(first)["runId"] == "shared-run"
    assert index.ready_manifest(second)["runId"] == "shared-run"
    assert index.open_artifact(first, "answer.bin").read() == b"first"
    assert index.open_artifact(second, "answer.bin").read() == b"second"


async def test_published_triple_survives_restart_and_rebuild(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    identity = results.ResultIdentity("bench", "job-1", "shared-run")

    await publish(index, identity, "answer.bin", b"published")
    del index
    restarted = ResultsIndex(tmp_path)
    restarted.rebuild()

    assert restarted.ready_manifest(identity)["runId"] == "shared-run"
    assert restarted.open_artifact(identity, "answer.bin").read() == b"published"


async def test_legacy_uid_identity_is_removed_without_blocking_new_results(
    tmp_path: Path,
) -> None:
    legacy_key = "59050385603661a295144e664a8edbcf2b11f3c301efef3092cfdefa86b5e7ba"
    legacy_identity = json.dumps(
        {
            "namespace": "bench",
            "jobId": "job-legacy",
            "runId": "run-legacy",
            "objectUid": "4f78fcbe-9aae-4cc9-ae19-204231b21575",
            "created": 100.0,
        },
        separators=(",", ":"),
    )
    for collection in (".staging", "runs"):
        legacy_run = tmp_path / collection / legacy_key
        legacy_run.mkdir(parents=True)
        (legacy_run / ".aiperf-result-identity.json").write_text(legacy_identity)
        (legacy_run / "legacy.bin").write_bytes(b"legacy")

    index = ResultsIndex(tmp_path)
    index.rebuild()

    assert index.stats()["stagingRuns"] == 0
    assert index.stats()["publishedRuns"] == 0
    assert not (tmp_path / ".staging" / legacy_key).exists()
    assert not (tmp_path / "runs" / legacy_key).exists()
    current = results.ResultIdentity("bench", "job-current", "run-current")
    await publish(index, current, "current.bin", b"current")
    assert index.open_artifact(current, "current.bin").read() == b"current"


async def test_replayed_artifact_after_publish_never_recreates_staging(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    identity_type = getattr(results, "ResultIdentity", None)
    assert identity_type is not None
    identity = identity_type("bench", "job-1", "run-1")
    body = b"published"
    await publish(index, identity, "answer.bin", body)

    assert not await index.stage_artifact(
        identity,
        "answer.bin",
        chunks(body),
        hashlib.sha256(body).hexdigest(),
        len(body),
    )
    assert index.staging_stats() == {"runs": 0, "bytes": 0}


async def test_staging_refuses_symlink_ancestor_without_writing_outside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / ".staging").symlink_to(outside, target_is_directory=True)
    with pytest.raises((UploadConflict, OSError)):
        ResultsIndex(root)
    assert list(outside.iterdir()) == []


async def test_staging_admission_is_bounded_and_expired_runs_are_collected(
    tmp_path: Path,
) -> None:
    now = [100.0]
    limits_type = getattr(results, "StorageLimits", None)
    assert limits_type is not None
    limits = limits_type(
        max_staging_runs=1,
        max_staging_bytes=8,
        max_run_bytes=8,
        max_artifacts_per_run=2,
        staging_ttl_seconds=10,
    )
    index = ResultsIndex(tmp_path, limits=limits, now=lambda: now[0])
    identity_type = getattr(results, "ResultIdentity", None)
    assert identity_type is not None
    first = identity_type("bench", "job-1", "run-1")
    second = identity_type("bench", "job-2", "run-2")
    assert await index.stage_artifact(
        first, "a.bin", chunks(b"12345678"), hashlib.sha256(b"12345678").hexdigest(), 8
    )
    with pytest.raises(UploadTooLarge):
        await index.stage_artifact(
            second, "b.bin", chunks(b"x"), hashlib.sha256(b"x").hexdigest(), 1
        )

    now[0] = 111.0
    assert await index.stage_artifact(
        second, "b.bin", chunks(b"x"), hashlib.sha256(b"x").hexdigest(), 1
    )
    assert index.staging_stats() == {"runs": 1, "bytes": 1}
    assert not any(entry.name.startswith(".upload-") for entry in os.scandir(tmp_path))


async def test_restart_orphan_temporary_bytes_remain_inside_global_quota(
    tmp_path: Path,
) -> None:
    limits = results.StorageLimits(
        max_staging_runs=2,
        max_staging_bytes=8,
        max_run_bytes=8,
        max_artifacts_per_run=3,
        staging_ttl_seconds=60,
    )
    index = ResultsIndex(tmp_path, limits=limits)
    first = results.ResultIdentity("bench", "job-1", "run-1")
    second = results.ResultIdentity("bench", "job-2", "run-2")
    assert await index.stage_artifact(
        first, "a.bin", chunks(b"x"), hashlib.sha256(b"x").hexdigest(), 1
    )
    staging_run = next((tmp_path / ".staging").iterdir())
    (staging_run / ".upload-crash-orphan").write_bytes(b"1234567")

    with pytest.raises(UploadTooLarge):
        await index.stage_artifact(
            second, "b.bin", chunks(b"y"), hashlib.sha256(b"y").hexdigest(), 1
        )


async def test_published_run_quota_is_bounded_and_expired_runs_are_collected(
    tmp_path: Path,
) -> None:
    now = [100.0]
    limits = results.StorageLimits(
        max_staging_runs=2,
        max_staging_bytes=16,
        max_run_bytes=8,
        max_artifacts_per_run=2,
        staging_ttl_seconds=60,
        max_published_runs=1,
        max_published_bytes=8,
        published_ttl_seconds=10,
    )
    index = ResultsIndex(tmp_path, limits=limits, now=lambda: now[0])
    first = results.ResultIdentity("bench", "job-1", "run-1")
    second = results.ResultIdentity("bench", "job-2", "run-2")
    await publish(index, first, "a.bin", b"12345678")
    body = b"y"
    assert await index.stage_artifact(
        second, "b.bin", chunks(body), hashlib.sha256(body).hexdigest(), len(body)
    )
    document = manifest(second.run_id, "b.bin", body)

    with pytest.raises(UploadTooLarge):
        index.commit_manifest(
            second,
            document,
            hashlib.sha256(document).hexdigest(),
            len(document),
        )

    now[0] = 111.0
    assert index.commit_manifest(
        second,
        document,
        hashlib.sha256(document).hexdigest(),
        len(document),
    )
    assert index.ready_manifest(first) is None
    assert index.ready_manifest(second) is not None


async def test_expiry_removes_published_bytes_without_tombstones(
    tmp_path: Path,
) -> None:
    now = [100.0]
    limits = results.StorageLimits(
        max_staging_runs=2,
        max_staging_bytes=32,
        max_run_bytes=16,
        max_artifacts_per_run=2,
        staging_ttl_seconds=60,
        max_published_runs=1,
        max_published_bytes=16,
        published_ttl_seconds=10,
    )
    index = ResultsIndex(tmp_path, limits=limits, now=lambda: now[0])
    expired = results.ResultIdentity("bench", "job-1", "run-1")
    current = results.ResultIdentity("bench", "job-2", "run-2")
    await publish(index, expired, "a.bin", b"expired")

    now[0] = 111.0
    await publish(index, current, "b.bin", b"current")

    restarted = ResultsIndex(tmp_path, limits=limits, now=lambda: now[0])
    restarted.rebuild()
    assert restarted.ready_manifest(expired) is None
    assert await restarted.stage_artifact(
        expired,
        "a.bin",
        chunks(b"expired"),
        hashlib.sha256(b"expired").hexdigest(),
        len(b"expired"),
    )
    assert not (tmp_path / ".expired").exists()
