# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for completion-path blocking-I/O off-loading and artifact byte totals.

Covers:
- ``_gather_index_inputs`` counting nested result files (``checkpoints/``)
- every artifact-validating helper running on a worker thread, never on the
  kopf event loop
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import orjson
import pytest
import zstandard
from pytest import param

from aiperf.kubernetes.crd_models import ControllerFetchResult
from aiperf.operator.handlers import completion
from aiperf.operator.results_layout import epoch_key_from_body, run_dir

_NAMESPACE = "aiperf-prod"
_JOB_ID = "aiperf-bench-9c31"
_UID = "6f1c2b04-8a5d-4d1b-9c7e-2f0a1b3c4d5e"
_BODY: dict[str, Any] = {
    "metadata": {
        "name": _JOB_ID,
        "namespace": _NAMESPACE,
        "uid": _UID,
        "creationTimestamp": "2024-04-25T17:02:03Z",
    },
    "spec": {},
}
_EPOCH = epoch_key_from_body(_BODY)

_SUMMARY = {
    "end_time": "2024-04-25T17:05:00Z",
    "metrics": {"request_count": {"avg": 3}},
}


def _write(base_dir: Path, relative_name: str, payload: bytes) -> Path:
    path = run_dir(base_dir, _NAMESPACE, _JOB_ID, _EPOCH) / relative_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _seed_run(base_dir: Path, *, nested: bool = False) -> None:
    _write(
        base_dir,
        "profile_export_aiperf.json",
        orjson.dumps(_SUMMARY),
    )
    _write(base_dir, "profile_export_aiperf.csv", b"metric,value\n")
    if nested:
        _write(base_dir, "checkpoints/shard-0.bin", b"x" * 512)
        _write(base_dir, "checkpoints/nested/shard-1.bin", b"y" * 256)


class TestGatherIndexInputsByteTotal:
    """``_gather_index_inputs`` must report the whole tree, not just its top level."""

    @pytest.mark.parametrize(
        "nested,extra_bytes",
        [
            param(False, 0, id="flat_only"),
            param(True, 512 + 256, id="with_subdirectories"),
        ],
    )  # fmt: skip
    def test_gather_index_inputs_nested_files_counted_in_total(
        self, tmp_path: Path, nested: bool, extra_bytes: int
    ) -> None:
        _seed_run(tmp_path, nested=nested)
        flat_bytes = len(orjson.dumps(_SUMMARY)) + len(b"metric,value\n")

        with mock_patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            DIR=tmp_path,
        ):
            _, _, end_time, total_size_bytes = completion._gather_index_inputs(
                _NAMESPACE, _JOB_ID, _EPOCH
            )

        assert end_time == _SUMMARY["end_time"]
        assert total_size_bytes == flat_bytes + extra_bytes

    def test_gather_index_inputs_compressed_summary_counts_nested_bytes(
        self, tmp_path: Path
    ) -> None:
        blob = zstandard.ZstdCompressor().compress(orjson.dumps(_SUMMARY))
        _write(tmp_path, "profile_export_aiperf.json.zst", blob)
        _write(tmp_path, "checkpoints/shard-0.bin", b"z" * 1024)

        with mock_patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            DIR=tmp_path,
        ):
            summary_blob, _, end_time, total_size_bytes = (
                completion._gather_index_inputs(_NAMESPACE, _JOB_ID, _EPOCH)
            )

        assert summary_blob == blob
        assert end_time == _SUMMARY["end_time"]
        assert total_size_bytes == len(blob) + 1024


# Every helper below reads (and for ``.zst`` decompresses) a key export, which
# must never happen inline on the single kopf event loop.
_OFFLOADED_HELPERS = (
    "_demote_missing_publication_artifacts",
    "_capture_publication_artifacts",
    "_record_results_on_status",
    "_gather_index_inputs",
    "_load_phase_manifest_payload",
    "_final_artifacts_intact",
    "_recover_result_from_disk",
    "_demote_unmaterialized_result_files",
    "_key_files_materialized",
)


def _thread_recording_patches(
    calls: dict[str, list[int]],
) -> list[Any]:
    """Patch each blocking helper with a delegating thread-ident recorder."""
    patches = []
    for name in _OFFLOADED_HELPERS:
        original = getattr(completion, name)

        def record(
            *args: Any, _original: Any = original, _name: str = name, **kwargs: Any
        ) -> Any:
            calls.setdefault(_name, []).append(threading.get_ident())
            return _original(*args, **kwargs)

        patches.append(mock_patch.object(completion, name, record))
    return patches


@pytest.mark.asyncio
async def test_apply_completion_results_blocking_helpers_run_off_event_loop(
    tmp_path: Path,
) -> None:
    _seed_run(tmp_path, nested=True)
    calls: dict[str, list[int]] = {}
    result = ControllerFetchResult(
        metrics={"metrics": {"request_count": {"avg": 3}}},
        downloaded=["profile_export_aiperf.json", "profile_export_aiperf.csv"],
        checkpoints=[],
        error="",
    )
    flags = completion._ResultFlags(
        has_metrics=True, has_files=True, has_error=False, success=True
    )

    with ExitStack() as stack:
        for patcher in (
            mock_patch(
                "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
                DIR=tmp_path,
                RETENTION_MAX_RUNS=0,
            ),
            mock_patch(
                "aiperf.operator.handlers.completion._parent_identity_is_current",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion._run_retention_pass",
                new=AsyncMock(),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion._update_job_index_safe",
                new=AsyncMock(return_value=True),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion.runs_index",
                new=MagicMock(begin_catalog_update=MagicMock(return_value=None)),
            ),
            *_thread_recording_patches(calls),
        ):
            stack.enter_context(patcher)
        await completion._apply_completion_results(
            body=_BODY,
            namespace=_NAMESPACE,
            jobset_name="js",
            job_id=_JOB_ID,
            result=result,
            sb=MagicMock(),
            status={},
            flags=flags,
        )

    loop_thread = threading.get_ident()
    assert {
        "_demote_missing_publication_artifacts",
        "_capture_publication_artifacts",
        "_record_results_on_status",
        "_gather_index_inputs",
        "_final_artifacts_intact",
    } <= set(calls)
    offending = {
        name: idents
        for name, idents in calls.items()
        if any(ident == loop_thread for ident in idents)
    }
    assert not offending, f"blocking helpers ran on the event loop: {offending}"


@pytest.mark.asyncio
async def test_verify_final_artifact_publication_validates_off_event_loop(
    tmp_path: Path,
) -> None:
    _seed_run(tmp_path)
    idents: list[int] = []

    def record(*args: Any, **kwargs: Any) -> bool:
        idents.append(threading.get_ident())
        return True

    fingerprint = (completion._KeyArtifactFingerprint(name="x", size=1, mtime_ns=1),)
    with (
        mock_patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            DIR=tmp_path,
        ),
        mock_patch.object(completion, "_final_artifacts_intact", record),
    ):
        flags, materialized = await completion._verify_final_artifact_publication(
            namespace=_NAMESPACE,
            job_id=_JOB_ID,
            epoch=_EPOCH,
            flags=completion._ResultFlags(
                has_metrics=True, has_files=True, has_error=False, success=True
            ),
            expected_fingerprint=fingerprint,
            key_names=completion.DEFAULT_KEY_EXPORT_NAMES,
            sb=MagicMock(),
        )

    assert materialized is True
    assert flags.success is True
    assert idents and threading.get_ident() not in idents


@pytest.mark.asyncio
async def test_emit_accepted_completion_events_uses_precomputed_materialization(
    tmp_path: Path,
) -> None:
    """The event emitter must not re-validate artifacts on the event loop."""
    _seed_run(tmp_path)
    result = ControllerFetchResult(
        metrics=None,
        downloaded=["profile_export_aiperf.json"],
        checkpoints=[],
        error="",
    )
    flags = completion._ResultFlags(
        has_metrics=False, has_files=True, has_error=False, success=True
    )

    with (
        mock_patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            DIR=tmp_path,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion._key_files_materialized"
        ) as materialized,
        mock_patch("aiperf.operator.handlers.completion.events") as events,
    ):
        completion._emit_accepted_completion_events(
            body=_BODY,
            namespace=_NAMESPACE,
            jobset_name="js",
            job_id=_JOB_ID,
            result=result,
            status_patch={},
            flags=flags,
            key_names=completion.DEFAULT_KEY_EXPORT_NAMES,
            duration_sec=1.0,
            files_materialized=False,
        )

    materialized.assert_not_called()
    events.results_stored.assert_not_called()
    events.completed.assert_called_once()
