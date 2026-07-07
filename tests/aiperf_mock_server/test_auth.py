# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import pytest
from aiperf_mock_server import __main__ as mock_main
from aiperf_mock_server import app as mock_app
from aiperf_mock_server.app import InferenceAuthMiddleware
from aiperf_mock_server.config import MockServerConfig, set_server_config
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request


def test_auth_defaults_disable_authentication() -> None:
    config = MockServerConfig()

    assert config.api_key is None
    assert config.auth_header_name == "Authorization"


def test_auth_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_SERVER_API_KEY", "secret-key")
    monkeypatch.setenv("MOCK_SERVER_AUTH_HEADER_NAME", "X-API-Key")

    config = MockServerConfig()

    assert config.api_key == "secret-key"
    assert config.auth_header_name == "X-API-Key"


def _auth_test_client(config: MockServerConfig) -> TestClient:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/v1/models")
    async def models() -> dict[str, list[object]]:
        return {"data": []}

    @app.post("/v1/chat/completions")
    async def chat(_: Request) -> dict[str, str]:
        return {"ok": "true"}

    return TestClient(InferenceAuthMiddleware(app, config))


def test_auth_middleware_allows_protected_path_when_api_key_unset() -> None:
    client = _auth_test_client(MockServerConfig(api_key=None))

    response = client.post("/v1/chat/completions")

    assert response.status_code == 200


def test_auth_middleware_rejects_missing_header_on_protected_path() -> None:
    client = _auth_test_client(MockServerConfig(api_key="secret-key"))

    response = client.post("/v1/chat/completions")

    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}


def test_auth_middleware_rejects_wrong_header_on_protected_path() -> None:
    client = _auth_test_client(MockServerConfig(api_key="secret-key"))

    response = client.post("/v1/chat/completions", headers={"Authorization": "wrong"})

    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}


def test_auth_middleware_accepts_configured_header_on_protected_path() -> None:
    client = _auth_test_client(
        MockServerConfig(api_key="secret-key", auth_header_name="X-API-Key")
    )

    response = client.post("/v1/chat/completions", headers={"X-API-Key": "secret-key"})

    assert response.status_code == 200


def test_auth_middleware_accepts_bearer_form_on_authorization_header() -> None:
    client = _auth_test_client(MockServerConfig(api_key="secret-key"))

    response = client.post(
        "/v1/chat/completions", headers={"Authorization": "Bearer secret-key"}
    )

    assert response.status_code == 200


def test_auth_middleware_rejects_bearer_form_on_custom_header() -> None:
    client = _auth_test_client(
        MockServerConfig(api_key="secret-key", auth_header_name="X-API-Key")
    )

    response = client.post(
        "/v1/chat/completions", headers={"X-API-Key": "Bearer secret-key"}
    )

    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}


def test_auth_middleware_rejects_bearer_form_with_wrong_key() -> None:
    client = _auth_test_client(MockServerConfig(api_key="secret-key"))

    response = client.post(
        "/v1/chat/completions", headers={"Authorization": "Bearer wrong"}
    )

    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}


def test_auth_middleware_leaves_non_inference_path_open() -> None:
    client = _auth_test_client(MockServerConfig(api_key="secret-key"))

    response = client.get("/health")

    assert response.status_code == 200


def test_serve_redacts_api_key_from_startup_logs(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mock_main.uvicorn, "run", lambda *args, **kwargs: None)

    with caplog.at_level(logging.INFO, logger="aiperf_mock_server.__main__"):
        mock_main.serve(MockServerConfig(api_key="secret-key"))

    assert "secret-key" not in caplog.text
    assert "api_key" not in caplog.text


def test_set_server_config_redacts_api_key_from_debug_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    original_config = mock_app.server_config
    original_env_value = os.environ.get("MOCK_SERVER_API_KEY")
    try:
        with caplog.at_level(logging.DEBUG, logger="aiperf_mock_server.config"):
            set_server_config(MockServerConfig(api_key="secret-key"))

        assert os.environ["MOCK_SERVER_API_KEY"] == "secret-key"
        assert "secret-key" not in caplog.text
        assert "MOCK_SERVER_API_KEY = <redacted>" in caplog.text
    finally:
        set_server_config(original_config)
        if original_env_value is None:
            os.environ.pop("MOCK_SERVER_API_KEY", None)
        else:
            os.environ["MOCK_SERVER_API_KEY"] = original_env_value


def test_public_health_and_root_do_not_expose_api_key() -> None:
    original_config = mock_app.server_config
    config = MockServerConfig(api_key="secret-key")
    set_server_config(config)
    mock_app.server_config = config
    try:
        client = TestClient(mock_app.app)

        health_response = client.get("/health")
        root_response = client.get("/")

        assert health_response.status_code == 200
        assert root_response.status_code == 200
        assert "api_key" not in health_response.json()["config"]
        assert "api_key" not in root_response.json()["config"]
    finally:
        set_server_config(original_config)
        mock_app.server_config = original_config


def test_auth_middleware_leaves_models_path_open() -> None:
    client = _auth_test_client(MockServerConfig(api_key="secret-key"))

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {"data": []}
