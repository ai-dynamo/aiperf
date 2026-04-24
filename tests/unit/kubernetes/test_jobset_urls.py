# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.jobset_urls.

Verifies URL construction for the JobSet CRD manifest and the install hint
printed by preflight. ``get_latest_jobset_version()`` hits GitHub and is
exercised with a stubbed aiohttp client.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import orjson
import pytest

from aiperf.kubernetes import jobset_urls


class TestManifestUrl:
    """URL builder for the JobSet release manifest."""

    def test_explicit_version_used_verbatim(self) -> None:
        url = jobset_urls.get_jobset_manifest_url("v0.7.1")
        assert "v0.7.1" in url
        assert url.endswith("/v0.7.1/manifests.yaml")

    def test_none_version_falls_back_to_pinned_version(self) -> None:
        url = jobset_urls.get_jobset_manifest_url(None)
        assert jobset_urls.JOBSET_FALLBACK_VERSION in url

    def test_url_is_https_github_release(self) -> None:
        """Downstream installers must always use HTTPS GitHub release URLs."""
        url = jobset_urls.get_jobset_manifest_url("v0.5.2")
        assert url.startswith("https://github.com/")
        assert "/releases/download/" in url
        assert jobset_urls.JOBSET_GITHUB_REPO in url


class TestInstallHint:
    """Printable hint string used in preflight output."""

    def test_hint_contains_kubectl_apply(self) -> None:
        hint = jobset_urls.get_jobset_install_hint()
        assert "kubectl apply" in hint
        assert "--server-side" in hint

    def test_hint_embeds_the_manifest_url(self) -> None:
        hint = jobset_urls.get_jobset_install_hint("v0.7.1")
        assert jobset_urls.get_jobset_manifest_url("v0.7.1") in hint

    def test_hint_with_no_version_uses_fallback_url(self) -> None:
        hint = jobset_urls.get_jobset_install_hint(None)
        assert jobset_urls.get_jobset_manifest_url(None) in hint


class TestLatestVersion:
    """Async GitHub API lookup for the latest release tag."""

    @pytest.mark.asyncio
    async def test_returns_tag_on_successful_response(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read = AsyncMock(return_value=orjson.dumps({"tag_name": "v0.7.1"}))
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            tag = await jobset_urls.get_latest_jobset_version()

        assert tag == "v0.7.1"

    @pytest.mark.asyncio
    async def test_returns_none_on_client_error(self) -> None:
        with patch("aiohttp.ClientSession", side_effect=aiohttp.ClientError("boom")):
            tag = await jobset_urls.get_latest_jobset_version()

        assert tag is None

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self) -> None:
        with patch("aiohttp.ClientSession", side_effect=TimeoutError()):
            tag = await jobset_urls.get_latest_jobset_version()

        assert tag is None

    @pytest.mark.asyncio
    async def test_returns_none_on_malformed_json(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read = AsyncMock(return_value=b"not json at all")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            tag = await jobset_urls.get_latest_jobset_version()

        assert tag is None

    @pytest.mark.asyncio
    async def test_returns_none_when_tag_name_is_non_string(self) -> None:
        """GitHub never returns a non-string tag_name in practice, but defend
        anyway — we don't want to propagate a typed-wrong value downstream."""
        mock_resp = MagicMock()
        mock_resp.read = AsyncMock(return_value=orjson.dumps({"tag_name": 42}))
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            tag = await jobset_urls.get_latest_jobset_version()

        assert tag is None
