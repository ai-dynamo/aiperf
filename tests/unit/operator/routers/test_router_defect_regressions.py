# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression locks for reviewer-accepted operator router fixes.

Each test here pins one defect that was fixed once and could silently
regress under a merge or rebase, because the fix is a one-line predicate or
a helper swap with no other caller asserting on it:

- ``jobs.event_sort_key`` parses timestamps instead of comparing ISO strings
- sweeps counter/index coercion degrades on a malformed aggregate/manifest
- ``results_analytics`` falls back to the ``job_spec.json.zst`` companion
- ``dashboard_proxy`` drops the upstream ``content-length``/``content-encoding``
- ``results_files_io`` allows nested files inside allowlisted subtrees
- the bundle generator does not await during teardown on client disconnect
- ``/jobs/{ns}/{name}/events`` and ``/logs`` validate their path params

Out of scope: the broader endpoint contracts, covered by the adversarial
suites in this directory.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import orjson
import pytest
import zstandard
from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from aiperf.operator.results_layout import run_dir
from aiperf.operator.routers.dashboard_proxy import _FORWARD_RESPONSE_HEADER_DROP
from aiperf.operator.routers.jobs import create_jobs_router, event_sort_key
from aiperf.operator.routers.jobs_models import EventEntry
from aiperf.operator.routers.results_analytics import _config_from_job_spec_file
from aiperf.operator.routers.results_files_io import (
    _serve_artifact_file,
    _stream_artifact_bundle,
)
from aiperf.operator.routers.sweeps import (
    _as_count,
    _as_index,
    _cells_from_aggregate,
    _children_manifest_from_doc,
)

_EPOCH = "1714150923"


# ============================================================
# jobs.event_sort_key
# ============================================================


def _entry(last_timestamp: str | None) -> EventEntry:
    return EventEntry(last_timestamp=last_timestamp)


class TestEventSortKey:
    """Events must order chronologically across mixed ISO-8601 renderings."""

    def test_zulu_and_offset_suffixes_compare_by_instant_not_string(self) -> None:
        # "2026-05-18T12:00:05Z" > "2026-05-18T12:00:10+00:00" lexicographically,
        # but is the OLDER instant. String sorting reversed these two.
        older = _entry("2026-05-18T12:00:05Z")
        newer = _entry("2026-05-18T12:00:10+00:00")

        entries = [older, newer]
        entries.sort(key=event_sort_key, reverse=True)

        assert entries == [newer, older]

    def test_naive_timestamp_is_treated_as_utc_and_stays_comparable(self) -> None:
        naive = _entry("2026-05-18T12:00:10")
        aware = _entry("2026-05-18T12:00:05Z")

        entries = [aware, naive]
        entries.sort(key=event_sort_key, reverse=True)

        assert entries == [naive, aware]

    @pytest.mark.parametrize("raw", [None, "", "not-a-timestamp"])
    def test_missing_or_unparseable_timestamps_sort_last(self, raw: str | None) -> None:
        bad = _entry(raw)
        good = _entry("2020-01-01T00:00:00Z")

        entries = [bad, good]
        entries.sort(key=event_sort_key, reverse=True)

        assert entries == [good, bad]
        assert event_sort_key(bad) == datetime.min.replace(tzinfo=UTC)


# ============================================================
# sweeps counter / index coercion
# ============================================================


class TestSweepCounterCoercion:
    """A malformed controller document must degrade, never 500."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (7, 7),
            (7.0, 7),
            (-1, 0),
            ("7", 0),
            ("abc", 0),
            (True, 0),
            (None, 0),
            ({}, 0),
        ],
    )
    def test_as_count_never_raises_and_never_goes_negative(
        self, raw: object, expected: int
    ) -> None:
        # Only JSON numbers coerce; numeric strings are controller-shape drift
        # and degrade to 0 rather than being silently trusted.
        assert _as_count(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (3, 3),
            (0, 0),
            (-1, None),
            ("3", None),
            ("abc", None),
            (True, None),
            (None, None),
        ],
    )
    def test_as_index_drops_unusable_values(
        self, raw: object, expected: int | None
    ) -> None:
        assert _as_index(raw) == expected

    def test_cells_from_aggregate_survives_non_numeric_counters(self) -> None:
        cells = _cells_from_aggregate(
            {
                "per_cell_aggregates": [
                    {
                        "variation_index": "not-a-number",
                        "variation_label": "concurrency=8",
                        "trials_completed": None,
                        "trials_failed": -4,
                        "children": [
                            {
                                "namespace": "bench",
                                "name": "child-0",
                                "trial_index": "oops",
                            }
                        ],
                    }
                ]
            }
        )

        assert len(cells) == 1
        assert cells[0].variation_index == 0
        assert cells[0].trials_completed == 0
        assert cells[0].trials_failed == 0
        assert cells[0].children[0].trial_index is None

    def test_children_manifest_survives_non_numeric_indices(self) -> None:
        manifest = _children_manifest_from_doc(
            {
                "sweep_run_epoch": _EPOCH,
                "children": [
                    {
                        "namespace": "bench",
                        "name": "child-0",
                        "variation_index": "x",
                        "trial_index": "y",
                    }
                ],
            },
            epoch=None,
        )

        assert manifest.children[0].variation_index == 0
        assert manifest.children[0].trial_index is None


# ============================================================
# results_analytics job_spec.json.zst fallback
# ============================================================


class TestJobSpecFileFallback:
    """With compress-on-disk the only spec on disk is the .zst companion."""

    @pytest.mark.asyncio
    async def test_zst_only_spec_is_decompressed_and_served(
        self, tmp_path: Path
    ) -> None:
        run = run_dir(tmp_path, "bench", "llama-load", _EPOCH)
        run.mkdir(parents=True)
        spec = {"benchmark": {"model": "llama-3-8b"}}
        (run / "job_spec.json.zst").write_bytes(
            zstandard.ZstdCompressor().compress(orjson.dumps(spec))
        )

        result = await _config_from_job_spec_file(
            tmp_path, "bench", "llama-load", _EPOCH
        )

        assert result is not None
        assert result["source"] == "file"
        assert result["spec"]["benchmark"]["model"] == "llama-3-8b"

    @pytest.mark.asyncio
    async def test_corrupt_zst_spec_returns_none_for_the_next_fallback(
        self, tmp_path: Path
    ) -> None:
        run = run_dir(tmp_path, "bench", "llama-load", _EPOCH)
        run.mkdir(parents=True)
        (run / "job_spec.json.zst").write_bytes(b"not-a-zstd-frame")

        assert (
            await _config_from_job_spec_file(tmp_path, "bench", "llama-load", _EPOCH)
            is None
        )

    @pytest.mark.asyncio
    async def test_raw_spec_still_wins_when_present(self, tmp_path: Path) -> None:
        run = run_dir(tmp_path, "bench", "llama-load", _EPOCH)
        run.mkdir(parents=True)
        (run / "job_spec.json").write_bytes(
            orjson.dumps({"benchmark": {"model": "raw"}})
        )
        (run / "job_spec.json.zst").write_bytes(
            zstandard.ZstdCompressor().compress(
                orjson.dumps({"benchmark": {"model": "zst"}})
            )
        )

        result = await _config_from_job_spec_file(
            tmp_path, "bench", "llama-load", _EPOCH
        )

        assert result is not None
        assert result["spec"]["benchmark"]["model"] == "raw"


# ============================================================
# dashboard_proxy response header hygiene
# ============================================================


class TestDashboardProxyResponseHeaderDrop:
    """aiohttp decodes the upstream body, so its framing headers must not survive."""

    @pytest.mark.parametrize(
        "header",
        ["content-length", "content-encoding", "transfer-encoding", "connection"],
    )
    def test_stale_framing_headers_are_dropped(self, header: str) -> None:
        assert header in _FORWARD_RESPONSE_HEADER_DROP

    def test_content_type_is_still_forwarded(self) -> None:
        assert "content-type" not in _FORWARD_RESPONSE_HEADER_DROP


# ============================================================
# results_files_io nested-artifact allowlist
# ============================================================


def _get_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "http_version": "1.1",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 0),
            "server": ("aiperf.operator.local", 80),
            "scheme": "http",
        }
    )


class TestNestedArtifactAllowlist:
    """The listing endpoint enumerates allowlisted subtrees recursively."""

    def test_nested_file_inside_allowlisted_subtree_is_servable(
        self, tmp_path: Path
    ) -> None:
        nested = tmp_path / "checkpoints" / "shard-0"
        nested.mkdir(parents=True)
        (nested / "ckpt.parquet").write_bytes(b"parquet-bytes")

        response = _serve_artifact_file(
            _get_request(),
            tmp_path,
            "checkpoints/shard-0/ckpt.parquet",
            allowed_relative_dirs=("checkpoints",),
        )

        assert response.status_code == 200

    def test_root_level_file_is_servable(self, tmp_path: Path) -> None:
        (tmp_path / "aggregate.json").write_bytes(b"{}")

        response = _serve_artifact_file(
            _get_request(),
            tmp_path,
            "aggregate.json",
            allowed_relative_dirs=("checkpoints",),
        )

        assert response.status_code == 200

    def test_file_outside_the_allowlisted_subtrees_stays_404(
        self, tmp_path: Path
    ) -> None:
        other = tmp_path / "private" / "deep"
        other.mkdir(parents=True)
        (other / "secret.json").write_bytes(b"{}")

        with pytest.raises(HTTPException) as excinfo:
            _serve_artifact_file(
                _get_request(),
                tmp_path,
                "private/deep/secret.json",
                allowed_relative_dirs=("checkpoints",),
            )

        assert excinfo.value.status_code == 404


# ============================================================
# bundle streaming teardown
# ============================================================


class TestBundleStreamTeardown:
    """The zip generator must not await while unwinding a GeneratorExit."""

    @pytest.mark.asyncio
    async def test_client_disconnect_mid_stream_closes_without_error(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "big.txt").write_bytes(b"y" * (1024 * 1024))
        (tmp_path / "summary.json").write_bytes(b'{"x":1}')

        gen = _stream_artifact_bundle(tmp_path)
        assert await gen.__anext__()

        # Awaiting zf.close() here used to raise
        # "async generator ignored GeneratorExit".
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_success_path_still_emits_the_central_directory(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "summary.json").write_bytes(b'{"x":1}')

        blob = b"".join([chunk async for chunk in _stream_artifact_bundle(tmp_path)])

        assert zipfile.ZipFile(io.BytesIO(blob)).namelist() == ["summary.json"]


# ============================================================
# jobs events/logs path-param validation
# ============================================================


@pytest.fixture
async def jobs_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    app.include_router(create_jobs_router([object()], tmp_path))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf.operator.local"
    ) as client:
        yield client


class TestJobDiagnosticsPathValidation:
    """``/events`` and ``/logs`` validate before any apiserver or PVC access."""

    @pytest.mark.asyncio
    async def test_events_rejects_invalid_namespace(
        self, jobs_client: httpx.AsyncClient
    ) -> None:
        response = await jobs_client.get("/api/v1/jobs/Bad_NS/llama-load/events")
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_events_rejects_invalid_name(
        self, jobs_client: httpx.AsyncClient
    ) -> None:
        response = await jobs_client.get("/api/v1/jobs/bench/Bad_Name/events")
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_logs_rejects_invalid_name(
        self, jobs_client: httpx.AsyncClient
    ) -> None:
        response = await jobs_client.get(
            "/api/v1/jobs/bench/Bad_Name/logs", params={"pod": "p-0"}
        )
        assert response.status_code == 400
