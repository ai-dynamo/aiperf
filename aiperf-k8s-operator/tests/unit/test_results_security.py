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
    first = identity_type("team-a", "job-a", "shared-run", "uid-a")
    second = identity_type("team-b", "job-b", "shared-run", "uid-b")
    await publish(index, first, "answer.bin", b"first")
    await publish(index, second, "answer.bin", b"second")

    assert index.ready_manifest(first)["runId"] == "shared-run"
    assert index.ready_manifest(second)["runId"] == "shared-run"
    assert index.open_artifact(first, "answer.bin").read() == b"first"
    assert index.open_artifact(second, "answer.bin").read() == b"second"


async def test_recreated_object_uid_cannot_read_or_adopt_previous_results(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    previous = results.ResultIdentity(
        "bench", "job-1", "shared-run", "11111111-1111-4111-8111-111111111111"
    )
    replacement = results.ResultIdentity(
        "bench", "job-1", "shared-run", "22222222-2222-4222-8222-222222222222"
    )

    await publish(index, previous, "answer.bin", b"previous")
    await publish(index, replacement, "answer.bin", b"replacement")

    assert index.open_artifact(previous, "answer.bin").read() == b"previous"
    assert index.open_artifact(replacement, "answer.bin").read() == b"replacement"


async def test_replayed_artifact_after_publish_never_recreates_staging(
    tmp_path: Path,
) -> None:
    index = ResultsIndex(tmp_path)
    identity_type = getattr(results, "ResultIdentity", None)
    assert identity_type is not None
    identity = identity_type("bench", "job-1", "run-1", "uid-1")
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
    first = identity_type("bench", "job-1", "run-1", "uid-1")
    second = identity_type("bench", "job-2", "run-2", "uid-2")
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
    first = results.ResultIdentity("bench", "job-1", "run-1", "uid-1")
    second = results.ResultIdentity("bench", "job-2", "run-2", "uid-2")
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
    first = results.ResultIdentity("bench", "job-1", "run-1", "uid-1")
    second = results.ResultIdentity("bench", "job-2", "run-2", "uid-2")
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
    with pytest.raises(results.ResultsExpired):
        index.ready_manifest(first)
    assert index.ready_manifest(second) is not None


async def test_expired_completed_identity_stays_gone_until_authority_release(
    tmp_path: Path,
) -> None:
    expired_type = getattr(results, "ResultsExpired", None)
    assert expired_type is not None
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
    expired = results.ResultIdentity("bench", "job-1", "run-1", "uid-old")
    current = results.ResultIdentity("bench", "job-2", "run-2", "uid-current")
    await publish(index, expired, "a.bin", b"expired")

    now[0] = 111.0
    await publish(index, current, "b.bin", b"current")

    restarted = ResultsIndex(tmp_path, limits=limits, now=lambda: now[0])
    restarted.rebuild()
    with pytest.raises(expired_type):
        restarted.ready_manifest(expired)
    with pytest.raises(expired_type):
        await restarted.stage_artifact(
            expired,
            "a.bin",
            chunks(b"expired"),
            hashlib.sha256(b"expired").hexdigest(),
            len(b"expired"),
        )

    assert restarted.release_identity(expired)
    assert await restarted.stage_artifact(
        expired,
        "a.bin",
        chunks(b"expired"),
        hashlib.sha256(b"expired").hexdigest(),
        len(b"expired"),
    )


async def test_authority_release_before_expiry_purges_exact_published_identity(
    tmp_path: Path,
) -> None:
    now = [100.0]
    limits = results.StorageLimits(published_ttl_seconds=10)
    identity = results.ResultIdentity("bench", "job-1", "run-1", "uid-deleted")
    index = ResultsIndex(tmp_path, limits=limits, now=lambda: now[0])
    await publish(index, identity, "answer.bin", b"published")

    assert index.release_identity(identity)
    now[0] = 111.0

    restarted = ResultsIndex(tmp_path, limits=limits, now=lambda: now[0])
    restarted.rebuild()
    assert restarted.ready_manifest(identity) is None
    assert list((tmp_path / "runs").iterdir()) == []
    assert list((tmp_path / ".expired").iterdir()) == []
