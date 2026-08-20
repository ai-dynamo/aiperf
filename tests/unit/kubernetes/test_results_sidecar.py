# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.results_sidecar.

Covers behavior not exercised by tests/unit/operator/test_results_server.py
(different module). Focus:
- Ready-marker gating for top-level artifacts
- Checkpoint directory bypass of the ready-marker gate
- Path-traversal rejection and reserved marker-name rejection
- Listing behavior before/after readiness and for nonexistent base dir
- File download content-encoding negotiation and response headers
- write_ready_marker side effects
- _safe_resolve / _is_checkpoint_path / ready_marker_path helpers
"""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path
from unittest.mock import patch

import httpx
import orjson
import pytest
from pytest import param

from aiperf.kubernetes.results_sidecar import (
    CHECKPOINTS_DIR_NAME,
    PROCESSING_MARKER_NAME,
    READY_MARKER_NAME,
    _is_checkpoint_path,
    _is_ready,
    _safe_resolve,
    checkpoints_dir,
    create_app,
    processing_marker_path,
    ready_marker_path,
    write_processing_marker,
    write_ready_marker,
)

# ============================================================
# Helpers
# ============================================================


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
async def client(base_dir: Path):
    """Create an httpx AsyncClient for the sidecar app."""
    app = create_app(base_dir)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _mark_ready(base_dir: Path, *, was_cancelled: bool = False) -> Path:
    return write_ready_marker(base_dir, was_cancelled=was_cancelled)


# ============================================================
# Path helpers
# ============================================================


class TestPathHelpers:
    """Verify the small path-derivation helpers."""

    def test_ready_marker_path(self, base_dir: Path) -> None:
        assert ready_marker_path(base_dir) == base_dir / READY_MARKER_NAME

    def test_checkpoints_dir(self, base_dir: Path) -> None:
        assert checkpoints_dir(base_dir) == base_dir / CHECKPOINTS_DIR_NAME


class TestSafeResolve:
    """Verify path traversal protection at the sidecar helper layer."""

    def test_valid_file(self, base_dir: Path) -> None:
        (base_dir / "a.json").write_bytes(b"{}")
        resolved = _safe_resolve(base_dir, "a.json")
        assert resolved is not None
        assert resolved == (base_dir / "a.json").resolve()

    @pytest.mark.parametrize(
        "filename",
        [
            param("../escape.txt", id="parent-traversal"),
            param("../../etc/passwd", id="multi-parent-traversal"),
            param("sub/../../out", id="nested-traversal"),
        ],
    )  # fmt: skip
    def test_blocks_traversal(self, base_dir: Path, filename: str) -> None:
        assert _safe_resolve(base_dir, filename) is None

    def test_null_byte_returns_none(self, base_dir: Path) -> None:
        assert _safe_resolve(base_dir, "a\x00b.json") is None


class TestIsReady:
    """Verify readiness detection based on the marker file."""

    def test_missing_marker_is_not_ready(self, base_dir: Path) -> None:
        assert _is_ready(base_dir) is False

    def test_present_marker_is_ready(self, base_dir: Path) -> None:
        _mark_ready(base_dir)
        assert _is_ready(base_dir) is True


class TestIsCheckpointPath:
    """Verify checkpoint-path detection for ready-gate bypass."""

    def test_checkpoint_file_detected(self, base_dir: Path) -> None:
        cp = base_dir / CHECKPOINTS_DIR_NAME
        cp.mkdir()
        path = cp / "cp1.json"
        path.write_bytes(b"{}")
        assert _is_checkpoint_path(base_dir.resolve(), path.resolve()) is True

    def test_top_level_file_not_checkpoint(self, base_dir: Path) -> None:
        path = base_dir / "metrics.json"
        path.write_bytes(b"{}")
        assert _is_checkpoint_path(base_dir.resolve(), path.resolve()) is False

    def test_outside_base_dir_not_checkpoint(
        self, base_dir: Path, tmp_path: Path
    ) -> None:
        other = tmp_path / "other.txt"
        other.write_bytes(b"x")
        assert _is_checkpoint_path(base_dir.resolve(), other.resolve()) is False


# ============================================================
# result transaction markers
# ============================================================


class TestWriteProcessingMarker:
    """Verify export begins by closing any readiness left by an older attempt."""

    def test_removes_stale_ready_before_installing_processing_marker(
        self, base_dir: Path
    ) -> None:
        write_ready_marker(base_dir)

        marker = write_processing_marker(base_dir)

        assert marker == processing_marker_path(base_dir)
        assert orjson.loads(marker.read_bytes()) == {"processing": True}
        assert not ready_marker_path(base_dir).exists()

    def test_processing_write_failure_leaves_stale_ready_removed(
        self, base_dir: Path
    ) -> None:
        write_ready_marker(base_dir)

        with (
            patch(
                "aiperf.common.results_markers._atomic_write_marker",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(OSError, match="disk full"),
        ):
            write_processing_marker(base_dir)

        assert not ready_marker_path(base_dir).exists()
        assert not processing_marker_path(base_dir).exists()

    def test_atomic_install_fsyncs_file_then_replaces_then_fsyncs_directory(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common import results_markers

        ready_marker_path(base_dir).write_bytes(b"stale")
        events: list[str] = []
        real_fsync = results_markers.os.fsync
        real_replace = results_markers.os.replace

        def tracking_fsync(file_descriptor: int) -> None:
            mode = os.fstat(file_descriptor).st_mode
            events.append("directory-fsync" if stat.S_ISDIR(mode) else "file-fsync")
            real_fsync(file_descriptor)

        def tracking_replace(source: Path, target: Path) -> None:
            events.append("replace")
            real_replace(source, target)

        monkeypatch.setattr(results_markers.os, "fsync", tracking_fsync)
        monkeypatch.setattr(results_markers.os, "replace", tracking_replace)

        write_processing_marker(base_dir)

        assert events == [
            "directory-fsync",
            "file-fsync",
            "replace",
            "directory-fsync",
        ]


class TestWriteReadyMarker:
    """Verify the marker writer produces expected on-disk state."""

    def test_creates_marker_file(self, base_dir: Path) -> None:
        path = write_ready_marker(base_dir)
        assert path == ready_marker_path(base_dir)
        assert path.is_file()

    def test_marker_contents_default(self, base_dir: Path) -> None:
        path = write_ready_marker(base_dir)
        data = orjson.loads(path.read_bytes())
        assert data == {"ready": True, "was_cancelled": False}

    def test_marker_contents_cancelled(self, base_dir: Path) -> None:
        path = write_ready_marker(base_dir, was_cancelled=True)
        data = orjson.loads(path.read_bytes())
        assert data == {"ready": True, "was_cancelled": True}

    def test_creates_base_dir_if_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "does_not_exist"
        assert not target.exists()
        path = write_ready_marker(target)
        assert path.is_file()
        assert target.is_dir()

    def test_ready_is_installed_before_processing_marker_is_cleared(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common import results_markers

        write_processing_marker(base_dir)
        real_clear = results_markers.clear_processing_marker

        def assert_ready_then_clear(directory: Path) -> None:
            assert ready_marker_path(directory).is_file()
            real_clear(directory)

        monkeypatch.setattr(
            results_markers, "clear_processing_marker", assert_ready_then_clear
        )

        write_ready_marker(base_dir)

        assert ready_marker_path(base_dir).is_file()
        assert not processing_marker_path(base_dir).exists()

    def test_atomic_replace_failure_keeps_processing_and_cleans_temporary(
        self, base_dir: Path
    ) -> None:
        write_processing_marker(base_dir)

        with (
            patch(
                "aiperf.kubernetes.results_markers.os.replace",
                side_effect=OSError("rename failed"),
            ),
            pytest.raises(OSError, match="rename failed"),
        ):
            write_ready_marker(base_dir)

        assert processing_marker_path(base_dir).is_file()
        assert not ready_marker_path(base_dir).exists()
        assert not list(base_dir.glob(f".{READY_MARKER_NAME}.*.tmp"))

    def test_directory_fsync_failure_after_replace_rolls_back_ready(
        self, base_dir: Path
    ) -> None:
        write_processing_marker(base_dir)

        with (
            patch(
                "aiperf.common.results_markers._fsync_directory",
                side_effect=OSError("directory fsync failed"),
            ),
            pytest.raises(OSError, match="directory fsync failed"),
        ):
            write_ready_marker(base_dir)

        assert not ready_marker_path(base_dir).exists()
        assert processing_marker_path(base_dir).is_file()

    def test_processing_clear_fsync_failure_rolls_back_ready_and_restores_processing(
        self, base_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common import results_markers

        write_processing_marker(base_dir)
        real_fsync_directory = results_markers._fsync_directory
        directory_fsync_calls = 0

        def fail_processing_clear(directory: Path) -> None:
            nonlocal directory_fsync_calls
            directory_fsync_calls += 1
            if directory_fsync_calls == 2:
                raise OSError("processing clear fsync failed")
            real_fsync_directory(directory)

        monkeypatch.setattr(results_markers, "_fsync_directory", fail_processing_clear)

        with pytest.raises(OSError, match="processing clear fsync failed"):
            write_ready_marker(base_dir)

        assert not ready_marker_path(base_dir).exists()
        assert processing_marker_path(base_dir).is_file()
        assert orjson.loads(processing_marker_path(base_dir).read_bytes()) == {
            "processing": True
        }


# ============================================================
# /healthz
# ============================================================


class TestHealthz:
    """Verify the sidecar health endpoint."""

    @pytest.mark.asyncio
    async def test_healthz_ok(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ============================================================
# /api/results/list
# ============================================================


class TestListEndpoint:
    """Verify /api/results/list gating and checkpoint handling."""

    @pytest.mark.asyncio
    async def test_list_not_ready_hides_top_level(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (base_dir / "metrics.json").write_bytes(b"{}")
        (base_dir / "profile.txt").write_bytes(b"x")

        resp = await client.get("/api/results/list")
        assert resp.status_code == 200
        assert resp.json()["files"] == []

    @pytest.mark.asyncio
    async def test_list_ready_exposes_top_level(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (base_dir / "metrics.json").write_bytes(b"{}")
        (base_dir / "profile.txt").write_bytes(b"xx")
        _mark_ready(base_dir)

        resp = await client.get("/api/results/list")
        names = {f["name"] for f in resp.json()["files"]}
        assert names == {"metrics.json", "profile.txt"}

    @pytest.mark.asyncio
    async def test_both_marker_crash_state_excludes_internal_markers(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (base_dir / "metrics.json").write_bytes(b"{}")
        write_processing_marker(base_dir)
        with patch("aiperf.kubernetes.results_markers.clear_processing_marker"):
            write_ready_marker(base_dir)

        assert ready_marker_path(base_dir).is_file()
        assert processing_marker_path(base_dir).is_file()

        resp = await client.get("/api/results/list")
        names = {f["name"] for f in resp.json()["files"]}
        assert names == {"metrics.json"}
        assert resp.json()["ready"] is True
        assert resp.json()["processing"] is True
        for marker_name in (READY_MARKER_NAME, PROCESSING_MARKER_NAME):
            marker_resp = await client.get(f"/api/results/files/{marker_name}")
            assert marker_resp.status_code == 400
            assert "reserved marker" in marker_resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_list_checkpoints_visible_when_not_ready(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        cp_dir = base_dir / CHECKPOINTS_DIR_NAME
        cp_dir.mkdir()
        (cp_dir / "cp0.json").write_bytes(b"{}")
        (cp_dir / "cp1.json").write_bytes(b"{}")

        resp = await client.get("/api/results/list")
        names = {f["name"] for f in resp.json()["files"]}
        assert names == {"checkpoints/cp0.json", "checkpoints/cp1.json"}

    @pytest.mark.asyncio
    async def test_list_checkpoints_recursive(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        cp_dir = base_dir / CHECKPOINTS_DIR_NAME / "nested"
        cp_dir.mkdir(parents=True)
        (cp_dir / "deep.json").write_bytes(b"{}")

        resp = await client.get("/api/results/list")
        names = {f["name"] for f in resp.json()["files"]}
        assert "checkpoints/nested/deep.json" in names

    @pytest.mark.asyncio
    async def test_list_sorted_by_name(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (base_dir / "b.txt").write_bytes(b"b")
        (base_dir / "a.txt").write_bytes(b"a")
        _mark_ready(base_dir)

        resp = await client.get("/api/results/list")
        names = [f["name"] for f in resp.json()["files"]]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_list_nonexistent_base_dir(self, tmp_path: Path) -> None:
        app = create_app(tmp_path / "does-not-exist")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.get("/api/results/list")
        assert resp.status_code == 200
        assert resp.json()["files"] == []

    @pytest.mark.asyncio
    async def test_list_recurses_into_subdirectories_when_ready(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        # Sweep aggregator writes nested artifacts under
        # ``/results/<ns>/sweeps/<name>/<epoch>/aggregate.json`` etc., so the
        # listing must recurse — a top-level-only walk would surface 0 files
        # for sweep-controller pods. Checkpoint subtree is still segmented
        # (covered by separate tests below).
        (base_dir / "metrics.json").write_bytes(b"{}")
        (base_dir / "extra").mkdir()
        (base_dir / "extra" / "nested.json").write_bytes(b"{}")
        _mark_ready(base_dir)

        resp = await client.get("/api/results/list")
        names = {f["name"] for f in resp.json()["files"]}
        assert names == {"metrics.json", "extra/nested.json"}

    @pytest.mark.asyncio
    async def test_list_reports_file_size(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (base_dir / "a.txt").write_bytes(b"x" * 128)
        _mark_ready(base_dir)

        resp = await client.get("/api/results/list")
        entry = next(f for f in resp.json()["files"] if f["name"] == "a.txt")
        assert entry["size"] == 128


# ============================================================
# /api/results/files/{filename}
# ============================================================


class TestDownloadEndpoint:
    """Verify /api/results/files/{filename} gating and traversal rejection."""

    @pytest.mark.asyncio
    async def test_download_top_level_not_ready_returns_404(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (base_dir / "metrics.json").write_bytes(b"{}")

        resp = await client.get("/api/results/files/metrics.json")
        assert resp.status_code == 404
        assert READY_MARKER_NAME in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_download_top_level_ready_succeeds(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        content = b'{"metric": 1}'
        (base_dir / "metrics.json").write_bytes(content)
        _mark_ready(base_dir)

        resp = await client.get(
            "/api/results/files/metrics.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert resp.status_code == 200
        assert resp.content == content
        assert resp.headers.get("x-filename") == "metrics.json"
        assert "metrics.json" in resp.headers.get("content-disposition", "")

    @pytest.mark.asyncio
    async def test_download_checkpoint_bypasses_ready_gate(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        cp_dir = base_dir / CHECKPOINTS_DIR_NAME
        cp_dir.mkdir()
        content = b'{"cp": true}'
        (cp_dir / "cp0.json").write_bytes(content)

        # No ready marker written — checkpoint must still be accessible
        resp = await client.get(
            "/api/results/files/checkpoints/cp0.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert resp.status_code == 200
        assert resp.content == content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename",
        [
            param("..%2F..%2Fetc%2Fpasswd", id="encoded-traversal"),
        ],
    )  # fmt: skip
    async def test_download_rejects_traversal(
        self, base_dir: Path, client: httpx.AsyncClient, filename: str
    ) -> None:
        _mark_ready(base_dir)
        resp = await client.get(f"/api/results/files/{filename}")
        # Either 400 (traversal detected) or 404 (not found). Never 200.
        assert resp.status_code in (400, 404)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "marker_name",
        [
            param(READY_MARKER_NAME, id="ready"),
            param(PROCESSING_MARKER_NAME, id="processing"),
        ],
    )  # fmt: skip
    async def test_download_rejects_marker_name(
        self, base_dir: Path, client: httpx.AsyncClient, marker_name: str
    ) -> None:
        _mark_ready(base_dir)
        resp = await client.get(f"/api/results/files/{marker_name}")
        assert resp.status_code == 400
        assert "reserved marker" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_download_missing_file_returns_404(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        _mark_ready(base_dir)
        resp = await client.get("/api/results/files/nope.json")
        assert resp.status_code == 404
        assert "Result file not found" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_download_missing_checkpoint_returns_404(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (base_dir / CHECKPOINTS_DIR_NAME).mkdir()

        resp = await client.get("/api/results/files/checkpoints/missing.json")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_download_sets_content_type_by_suffix(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (base_dir / "data.csv").write_bytes(b"a,b\n1,2\n")
        (base_dir / "info.json").write_bytes(b"{}")
        (base_dir / "note.txt").write_bytes(b"hello")
        _mark_ready(base_dir)

        # FastAPI's StreamingResponse sets media_type on content-type header.
        for name, expected in [
            ("data.csv", "text/csv"),
            ("info.json", "application/json"),
            ("note.txt", "text/plain"),
        ]:
            resp = await client.get(
                f"/api/results/files/{name}",
                headers={"Accept-Encoding": "identity"},
            )
            assert resp.status_code == 200
            assert resp.headers.get("content-type", "").startswith(expected)

    @pytest.mark.asyncio
    async def test_download_unknown_suffix_is_octet_stream(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (base_dir / "blob.bin").write_bytes(b"\x00\x01\x02")
        _mark_ready(base_dir)

        resp = await client.get(
            "/api/results/files/blob.bin",
            headers={"Accept-Encoding": "identity"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith(
            "application/octet-stream"
        )

    @pytest.mark.asyncio
    async def test_download_gzip_content_encoding(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        content = b"x" * 512
        (base_dir / "a.txt").write_bytes(content)
        _mark_ready(base_dir)

        resp = await client.get(
            "/api/results/files/a.txt",
            headers={"Accept-Encoding": "gzip"},
        )
        assert resp.status_code == 200
        # httpx transparently decompresses; content still matches source.
        assert resp.content == content

    @pytest.mark.asyncio
    async def test_download_identity_has_no_content_encoding(
        self, base_dir: Path, client: httpx.AsyncClient
    ) -> None:
        (base_dir / "a.txt").write_bytes(b"hi")
        _mark_ready(base_dir)

        resp = await client.get(
            "/api/results/files/a.txt",
            headers={"Accept-Encoding": "identity"},
        )
        # Identity encoding should not emit a Content-Encoding header.
        assert "content-encoding" not in {k.lower() for k in resp.headers}


# ============================================================
# create_app defaults
# ============================================================


class TestCreateApp:
    """Verify app factory argument handling."""

    def test_create_app_default_uses_env_dir(self) -> None:
        # Should construct without raising even when the default RESULTS_DIR
        # doesn't exist on this host.
        app = create_app()
        assert app.title == "AIPerf Controller Results Sidecar"

    @pytest.mark.asyncio
    async def test_create_app_override_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "alt"
        d.mkdir()
        (d / "only.txt").write_bytes(b"hi")
        write_ready_marker(d)

        app = create_app(d)
        transport = httpx.ASGITransport(app=app)
        async with asyncio.timeout(5):
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                resp = await c.get("/api/results/list")
        names = {f["name"] for f in resp.json()["files"]}
        assert names == {"only.txt"}
