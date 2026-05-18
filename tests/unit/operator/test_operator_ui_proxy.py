# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the local operator UI proxy development utility."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tools.operator_ui_proxy import create_app


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
