# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the local operator UI proxy development utility."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tools.operator_ui_proxy import (
    _collect_ui_file_mtimes,
    _parse_args,
    _scan_ui_files_once,
    create_app,
)


@pytest.mark.asyncio
async def test_live_serves_index_with_no_store_cache(tmp_path: Path) -> None:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text("<div>live ui</div>", encoding="utf-8")

    app = create_app(ui_dir=ui_dir, upstream="http://127.0.0.1:1", snapshots_dir=None)
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/live/")
        body = await response.text()

    assert response.status == 200
    assert body == "<div>live ui</div>"
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.asyncio
async def test_live_index_injects_reload_script_when_watch_enabled(
    tmp_path: Path,
) -> None:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text(
        "<html><body><div id='app'></div></body></html>", encoding="utf-8"
    )

    app = create_app(
        ui_dir=ui_dir,
        upstream="http://127.0.0.1:1",
        snapshots_dir=None,
        dev_reload=True,
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/live/")
        body = await response.text()

    assert response.status == 200
    assert "/__dev_reload/events" in body
    assert "new EventSource" in body
    assert "window.location.reload()" in body


@pytest.mark.asyncio
async def test_live_static_file_does_not_inject_reload_script(tmp_path: Path) -> None:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text(
        "<html><body>index</body></html>", encoding="utf-8"
    )
    (ui_dir / "app.js").write_text("console.log('app');", encoding="utf-8")

    app = create_app(
        ui_dir=ui_dir,
        upstream="http://127.0.0.1:1",
        snapshots_dir=None,
        dev_reload=True,
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/live/app.js")
        body = await response.text()

    assert response.status == 200
    assert body == "console.log('app');"
    assert "/__dev_reload/events" not in body


@pytest.mark.asyncio
async def test_dev_reload_events_emit_when_generation_changes(tmp_path: Path) -> None:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text(
        "<html><body>index</body></html>", encoding="utf-8"
    )

    app = create_app(
        ui_dir=ui_dir,
        upstream="http://127.0.0.1:1",
        snapshots_dir=None,
        dev_reload=True,
    )

    async with TestClient(TestServer(app)) as client:
        response = await asyncio.wait_for(client.get("/__dev_reload/events"), timeout=1)
        await asyncio.sleep(0.05)
        await _scan_ui_files_once(app, _collect_ui_file_mtimes(ui_dir))
        (ui_dir / "app.css").write_text("body { color: red; }", encoding="utf-8")
        await _scan_ui_files_once(app, _collect_ui_file_mtimes(ui_dir))
        body = await asyncio.wait_for(response.content.read(), timeout=1)

    assert response.status == 200
    assert body.decode("utf-8") == "event: reload\ndata: changed\n\n"
    assert response.content.at_eof()


@pytest.mark.asyncio
async def test_dev_reload_events_broadcast_one_change_to_two_clients(
    tmp_path: Path,
) -> None:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text(
        "<html><body>index</body></html>", encoding="utf-8"
    )

    app = create_app(
        ui_dir=ui_dir,
        upstream="http://127.0.0.1:1",
        snapshots_dir=None,
        dev_reload=True,
    )

    async with TestClient(TestServer(app)) as client:
        first_request = asyncio.create_task(client.get("/__dev_reload/events"))
        second_request = asyncio.create_task(client.get("/__dev_reload/events"))
        first_response = await asyncio.wait_for(first_request, timeout=1)
        second_response = await asyncio.wait_for(second_request, timeout=1)
        await asyncio.sleep(0.05)

        await _scan_ui_files_once(app, _collect_ui_file_mtimes(ui_dir))
        (ui_dir / "app.js").write_text("console.log('changed');", encoding="utf-8")
        await _scan_ui_files_once(app, _collect_ui_file_mtimes(ui_dir))

        first_body = await asyncio.wait_for(first_response.content.read(), timeout=1)
        second_body = await asyncio.wait_for(second_response.content.read(), timeout=1)

    assert first_response.status == 200
    assert second_response.status == 200
    assert first_body.decode("utf-8") == "event: reload\ndata: changed\n\n"
    assert second_body.decode("utf-8") == "event: reload\ndata: changed\n\n"
    assert first_response.content.at_eof()
    assert second_response.content.at_eof()


@pytest.mark.asyncio
async def test_dev_reload_events_returns_not_found_when_reload_disabled(
    tmp_path: Path,
) -> None:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text(
        "<html><body>index</body></html>", encoding="utf-8"
    )

    app = create_app(
        ui_dir=ui_dir,
        upstream="http://127.0.0.1:1",
        snapshots_dir=None,
        dev_reload=False,
    )

    async with TestClient(TestServer(app)) as client:
        response = await asyncio.wait_for(client.get("/__dev_reload/events"), timeout=1)

    assert response.status == 404


@pytest.mark.asyncio
async def test_dev_reload_file_scan_ignores_txt_and_notifies_for_ui_assets(
    tmp_path: Path,
) -> None:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text(
        "<html><body>index</body></html>", encoding="utf-8"
    )
    ignored = ui_dir / "notes.txt"
    ignored.write_text("first", encoding="utf-8")

    app = create_app(
        ui_dir=ui_dir,
        upstream="http://127.0.0.1:1",
        snapshots_dir=None,
        dev_reload=True,
    )
    mtimes = _collect_ui_file_mtimes(ui_dir)

    ignored.write_text("changed", encoding="utf-8")
    assert await _scan_ui_files_once(app, mtimes) is False

    (ui_dir / "styles.css").write_text("body { color: blue; }", encoding="utf-8")
    assert await _scan_ui_files_once(app, mtimes) is True

    assert await _scan_ui_files_once(app, mtimes) is False

    (ui_dir / "app.js").write_text("console.log('changed');", encoding="utf-8")
    assert await _scan_ui_files_once(app, mtimes) is True


@pytest.mark.asyncio
async def test_dev_reload_app_state_uses_single_appkey_without_string_duplicates(
    tmp_path: Path,
) -> None:
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text(
        "<html><body>index</body></html>", encoding="utf-8"
    )

    app = create_app(
        ui_dir=ui_dir,
        upstream="http://127.0.0.1:1",
        snapshots_dir=None,
        dev_reload=True,
    )

    assert "dev_reload_condition" not in app
    assert "dev_reload_generation" not in app


@pytest.mark.parametrize(
    "argv, expected_dev_reload",
    [
        pytest.param([], False, id="default-disabled"),
        pytest.param(["--dev-reload"], True, id="flag-enabled"),
    ],
)
def test_parse_args_accepts_dev_reload_flag(
    argv: list[str],
    expected_dev_reload: bool,
) -> None:
    args = _parse_args(argv)

    assert args.dev_reload is expected_dev_reload
