# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the static file router (index, dashboard, path traversal)."""

from unittest.mock import AsyncMock, patch

import pytest
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.api_service import FastAPIService
from aiperf.api.routers.static import _read_static


class TestStaticFileServing:
    """Test static file serving with path traversal protection."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename",
        [
            param("../secret.txt", id="parent-dir"),
            param("../../etc/passwd", id="etc-passwd"),
            param("static/../../../secret.txt", id="nested-traversal"),
            param("foo/../../../etc/passwd", id="deep-traversal"),
        ],
    )  # fmt: skip
    async def test_path_traversal_blocked(self, filename: str) -> None:
        """Test that path traversal attempts are blocked with 400."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await _read_static(filename)
        assert exc_info.value.status_code == 400
        assert "Invalid filename" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_404(self) -> None:
        """Test that non-existent files return 404."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await _read_static("nonexistent.html")
        assert exc_info.value.status_code == 404


class TestStaticPageEndpoints:
    """Test the static page serving endpoints."""

    def test_index_page_returns_html(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test index page serves HTML."""
        with patch(
            "aiperf.api.routers.static._read_static",
            new_callable=AsyncMock,
            return_value="<html>Index</html>",
        ):
            response = api_test_client.get("/")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_dashboard_page_returns_html(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test dashboard page serves HTML."""
        with patch(
            "aiperf.api.routers.static._read_static",
            new_callable=AsyncMock,
            return_value="<html>Dashboard</html>",
        ):
            response = api_test_client.get("/dashboard")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]


class TestDashboardLiveContract:
    """Integration-style tests that serve the real dashboard.html and validate
    the fields the inline JS (`renderConfig`) actually reads against a real
    BenchmarkConfig dump.
    """

    def test_dashboard_inline_js_has_updated_config_shape(
        self, api_test_client: TestClient
    ) -> None:
        """Regression: the /dashboard response must include the post-fix paths
        (cfg.models?.items, cfg.phases, connectionEpoch). If someone ships an
        older copy of dashboard.html the old loadgen/input paths would come
        back and the config bar silently breaks again.
        """
        resp = api_test_client.get("/dashboard")
        assert resp.status_code == 200
        html = resp.text
        assert "cfg.models?.items" in html
        assert "cfg.phases" in html
        assert "connectionEpoch" in html
        # Paths that belonged to the obsolete schema must stay gone.
        assert "cfg.loadgen" not in html
        assert "cfg.input" not in html
        assert "ep.model_names" not in html

    def test_api_config_shape_matches_renderConfig_expectations(
        self, api_test_client: TestClient
    ) -> None:
        """Contract: what /api/config returns must line up with the paths
        renderConfig reads from. If the backend response drifts, the
        dashboard's configuration strip silently empties out.
        """
        resp = api_test_client.get("/api/config")
        assert resp.status_code == 200
        cfg = resp.json()

        # Top-level keys renderConfig walks.
        assert "models" in cfg
        assert "endpoint" in cfg
        assert "phases" in cfg

        # models.items[*].name
        items = cfg["models"].get("items", [])
        assert items, "models.items must be populated"
        assert all("name" in it for it in items), (
            f"every models.items entry must have `name`; got {items!r}"
        )

        # endpoint.{type,urls}; api_key must be excluded.
        ep = cfg["endpoint"]
        assert ep.get("urls"), "endpoint.urls must be present"
        assert "api_key" not in ep, (
            f"api_key must be excluded from /api/config (found {ep.get('api_key')!r})"
        )

        # phases is a dict of phase-name -> phase config; every phase has `type`.
        assert isinstance(cfg["phases"], dict) and cfg["phases"]
        for phase_name, phase in cfg["phases"].items():
            assert "type" in phase, (
                f"phase {phase_name!r} missing `type`; renderConfig would skip it"
            )

    def test_dashboard_worker_status_whitelist_closed(
        self, api_test_client: TestClient
    ) -> None:
        """Regression for the CSS class-injection fix: the inline JS must
        whitelist worker statuses so a malformed enum value can't leak a
        stray class token.
        """
        resp = api_test_client.get("/dashboard")
        html = resp.text
        assert "KNOWN_STATUSES" in html
        # Guard against regression to the escapeHtml-only version.
        assert "worker-status ${escapeHtml(w.status" not in html
