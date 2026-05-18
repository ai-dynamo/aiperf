# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import orjson
import pytest

from aiperf.common.models.model_autodetect import autodetect_names


class _FakeRecord:
    def __init__(self, *, status: int | None, body_text: str) -> None:
        self.status = status
        resp = type("_Resp", (), {"text": body_text})()
        self.responses = [resp]


class _FakeClient:
    """Replays a fixed sequence of (status, body) responses."""

    def __init__(self, *, responses: list[tuple[int | None, str]]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.closed = False

    async def get_request(
        self, url: str, headers: dict[str, str], **_: Any
    ) -> _FakeRecord:
        self.urls.append(url)
        self.headers.append(headers)
        status, body = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return _FakeRecord(status=status, body_text=body)

    async def close(self) -> None:
        self.closed = True


def _install_fake_aiohttp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: list[tuple[int | None, str]],
) -> _FakeClient:
    fake = _FakeClient(responses=responses)

    def _factory(*_: Any, **__: Any) -> _FakeClient:
        return fake

    monkeypatch.setattr(
        "aiperf.common.models.model_autodetect.AioHttpClient",
        _factory,
    )
    return fake


def _models_body(*ids: str) -> str:
    return orjson.dumps({"data": [{"id": mid} for mid in ids]}).decode("utf-8")


def test_autodetect_picks_first_id_from_data(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    caplog.set_level(logging.WARNING, logger="aiperf.common.models.model_autodetect")
    fake = _install_fake_aiohttp(
        monkeypatch, responses=[(200, _models_body("model-a", "model-b"))]
    )

    result = asyncio.run(
        autodetect_names(
            urls=["http://localhost:8000"],
            headers={"Authorization": "Bearer token"},
            timeout_s=10.0,
        )
    )

    assert result == ["model-a"]
    assert fake.urls == ["http://localhost:8000/v1/models"]
    assert fake.headers[0]["Authorization"] == "Bearer token"
    assert fake.closed is True
    assert "2 models returned" in caplog.text
    assert "pass --model" in caplog.text
    assert "first listed model 'model-a'" in caplog.text


def test_autodetect_single_model_logs_info_not_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="aiperf.common.models.model_autodetect")
    _install_fake_aiohttp(monkeypatch, responses=[(200, _models_body("only-one"))])

    asyncio.run(
        autodetect_names(
            urls=["http://localhost:8000"],
            headers={},
            timeout_s=10.0,
        )
    )

    assert "Auto-detected model 'only-one'" in caplog.text
    assert "pass --model" not in caplog.text


def test_autodetect_timeout_exhausted_on_non_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-200 responses trigger retries until timeout -> TimeoutError."""
    _install_fake_aiohttp(monkeypatch, responses=[(404, "not found")] * 100)

    with pytest.raises(TimeoutError, match="Timed out"):
        asyncio.run(
            autodetect_names(
                urls=["http://localhost:8000"],
                headers={},
                timeout_s=0.01,
                interval_s=0.001,
            )
        )


def test_autodetect_retries_on_non_200_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server returns 503 on first attempt, then 200 with a model -- should succeed."""
    fake = _install_fake_aiohttp(
        monkeypatch,
        responses=[
            (503, "not ready"),
            (200, _models_body("my-model")),
        ],
    )

    result = asyncio.run(
        autodetect_names(
            urls=["http://localhost:8000"],
            headers={},
            timeout_s=30.0,
            interval_s=0.001,
        )
    )

    assert result == ["my-model"]
    assert len(fake.urls) == 2


def test_autodetect_retries_on_empty_data_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server returns 200 but empty data[] on first attempt, model appears next."""
    fake = _install_fake_aiohttp(
        monkeypatch,
        responses=[
            (200, orjson.dumps({"data": []}).decode("utf-8")),
            (200, _models_body("appeared-model")),
        ],
    )

    result = asyncio.run(
        autodetect_names(
            urls=["http://localhost:8000"],
            headers={},
            timeout_s=30.0,
            interval_s=0.001,
        )
    )

    assert result == ["appeared-model"]
    assert len(fake.urls) == 2


def test_autodetect_raises_valueerror_on_empty_urls() -> None:
    with pytest.raises(ValueError, match="at least one --url"):
        asyncio.run(autodetect_names(urls=[], headers={}))
