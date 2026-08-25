# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refusal classification and request-control regressions for results_operator.

Two shapes are covered:

- ``_collect_downloads`` classified every non-download against the job-level
  leaf policy, so a sweep artifact refused by ``_safe_sweep_artifact_path`` for
  a dotted *directory* component was reported as a download failure instead of
  a by-policy skip;
- the ``TypeError`` fallbacks in the no-redirect and request-timeout helpers
  applied to real ``aiohttp.ClientSession`` instances, where retrying without
  the kwarg would silently drop the control it enforces.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from pytest import param

from aiperf.kubernetes import results_operator as sweeps
from aiperf.kubernetes.results_operator import (
    _collect_downloads,
    _get_no_redirects,
    _get_with_request_timeout,
    _is_refused_name,
    _is_refused_sweep_name,
    _safe_sweep_artifact_path,
)

# ============================================================
# Refusal policy parity
# ============================================================


class TestRefusalPolicyMatchesResolution:
    """Each downloader's skip predicate matches the path policy it enforces."""

    @pytest.mark.parametrize(
        "display_name",
        [
            param("sweep_aggregate/.hidden/summary.json", id="dotted-mid-directory"),
            param(".hidden/summary.json", id="dotted-leading-directory"),
            param("a/.b/c/summary.json", id="dotted-deep-directory"),
            param("aggregate/.hidden.json", id="dotted-leaf"),
            param("../aggregate.json", id="parent-traversal"),
            param("/tmp/aggregate.json", id="absolute-path"),
            param("", id="empty-name"),
        ],
    )  # fmt: skip
    def test_sweep_refusal_predicate_agrees_with_safe_path(
        self, tmp_path: Path, display_name: str
    ) -> None:
        assert _is_refused_sweep_name(display_name) is True
        assert _safe_sweep_artifact_path(tmp_path, display_name) is None

    @pytest.mark.parametrize(
        "display_name",
        [
            param("summary.json", id="leaf"),
            param("sweep_aggregate/summary.json", id="nested"),
        ],
    )  # fmt: skip
    def test_sweep_refusal_predicate_allows_plain_paths(
        self, tmp_path: Path, display_name: str
    ) -> None:
        assert _is_refused_sweep_name(display_name) is False
        assert _safe_sweep_artifact_path(tmp_path, display_name) == (
            tmp_path / display_name
        )

    def test_job_policy_still_permits_dotted_directories(self) -> None:
        """The job-level policy is deliberately leaf-only; do not tighten it here."""
        assert _is_refused_name("checkpoints/.hidden/records-0.parquet") is False
        assert _is_refused_sweep_name("checkpoints/.hidden/records-0.parquet") is True


# ============================================================
# Skip-versus-failure classification
# ============================================================


class TestCollectDownloadsClassification:
    """Refused names are skips; only genuine non-deliveries are failures."""

    @pytest.mark.asyncio
    async def test_custom_predicate_marks_refused_names_as_skips(self) -> None:
        async def _download(file_info: dict) -> tuple[str, int] | None:
            return ("ok.json", 10) if file_info["name"] == "ok.json" else None

        outcome = await _collect_downloads(
            [
                {"name": "ok.json"},
                {"name": "sweep_aggregate/.hidden/summary.json"},
                {"name": "children.json"},
            ],
            _download,
            is_refused=_is_refused_sweep_name,
        )

        assert outcome.downloaded == [("ok.json", 10)]
        assert outcome.failed == ["children.json"]
        assert outcome.complete is False

    @pytest.mark.asyncio
    async def test_sweep_download_skips_dotted_directory_artifact(
        self, tmp_path: Path
    ) -> None:
        listing = [
            {"name": "aggregate.json"},
            {"name": "sweep_aggregate/.hidden/summary.json"},
        ]

        async def _download(_session, *, file_info, **_kwargs):
            name = file_info["name"]
            return (name, 10) if name == "aggregate.json" else None

        with (
            patch.object(
                sweeps, "_list_sweep_operator_files", AsyncMock(return_value=listing)
            ),
            patch.object(
                sweeps,
                "_download_sweep_operator_file",
                AsyncMock(side_effect=_download),
            ),
        ):
            outcome = await sweeps._download_all_sweep_operator_files(
                api_base="http://x",
                namespace="ns",
                sweep_name="s",
                output_dir=tmp_path,
                run="100",
            )

        assert outcome is not None
        assert outcome.downloaded == [("aggregate.json", 10)]
        assert outcome.failed == []
        assert outcome.complete is True


# ============================================================
# Request-control fallbacks
# ============================================================


class _TypeErrorSession:
    """Test double whose ``get`` rejects the control kwargs."""

    def __init__(self, message: str) -> None:
        self._message = message
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs: object) -> str:
        if kwargs:
            self.calls.append(dict(kwargs))
            raise TypeError(self._message)
        self.calls.append({})
        return url


class _RejectingClientSession(aiohttp.ClientSession):
    """Real client whose ``get`` raises the same shape of ``TypeError``."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def get(self, url: str, **kwargs: object) -> str:  # type: ignore[override]
        raise TypeError(self._message)


class TestRequestControlFallbacks:
    """Test-double fallbacks must never relax controls on a real client."""

    def test_no_redirect_fallback_applies_to_test_double(self) -> None:
        session = _TypeErrorSession(
            "get() got an unexpected keyword argument 'allow_redirects'"
        )

        assert _get_no_redirects(session, "http://x") == "http://x"
        assert session.calls == [{"allow_redirects": False}, {}]

    def test_timeout_fallback_applies_to_test_double(self) -> None:
        session = _TypeErrorSession(
            "get() got an unexpected keyword argument 'timeout'"
        )

        assert _get_with_request_timeout(session, "http://x") == "http://x"
        assert list(session.calls[0]) == ["timeout"]
        assert session.calls[1] == {}

    def test_unrelated_type_error_still_propagates(self) -> None:
        session = _TypeErrorSession("unhashable type: 'dict'")

        with pytest.raises(TypeError, match="unhashable"):
            _get_no_redirects(session, "http://x")

    @pytest.mark.asyncio
    async def test_no_redirect_does_not_retry_on_real_client_session(self) -> None:
        session = _RejectingClientSession(
            "get() got an unexpected keyword argument 'allow_redirects'"
        )
        try:
            with pytest.raises(TypeError, match="allow_redirects"):
                _get_no_redirects(session, "http://x")
        finally:
            await session.close()

    @pytest.mark.asyncio
    async def test_timeout_does_not_retry_on_real_client_session(self) -> None:
        session = _RejectingClientSession(
            "get() got an unexpected keyword argument 'timeout'"
        )
        try:
            with pytest.raises(TypeError, match="timeout"):
                _get_with_request_timeout(session, "http://x")
        finally:
            await session.close()
