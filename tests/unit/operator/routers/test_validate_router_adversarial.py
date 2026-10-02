# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for ``POST /api/v1/validate``.

The route is mounted without ``mutating_dependencies`` (see
``results_server.create_app``), so it is reachable unauthenticated and its whole
purpose is turning a hostile manifest into a structured response. Focuses on:
- manifests whose ``connectionsPerWorker`` used to crash the worker-count
  arithmetic, which surfaced as HTTP 500 on this route
- the response contract holding even when the validator itself raises

Out of scope: the validation rules themselves, covered by
``tests/unit/kubernetes/test_validate.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import param

from aiperf.kubernetes.cr_refs import AIPERF_API_VERSION
from aiperf.operator.routers import validate as mod
from aiperf.operator.routers.validate import create_validate_router


def _valid_manifest() -> dict[str, Any]:
    """Minimal AIPerfJob manifest that validates clean."""
    return {
        "apiVersion": AIPERF_API_VERSION,
        "kind": "AIPerfJob",
        "metadata": {"name": "my-bench"},
        "spec": {
            "benchmark": {
                "models": ["meta-llama/Llama-3.1-8B-Instruct"],
                "endpoint": {"urls": ["http://svc.ns.svc.cluster.local:8000"]},
                "datasets": [
                    {
                        "name": "default",
                        "type": "synthetic",
                        "entries": 100,
                        "prompts": {"isl": 128, "osl": 64},
                    }
                ],
                "phases": [
                    {
                        "name": "default",
                        "type": "concurrency",
                        "kind": "profiling",
                        "concurrency": 16,
                        "requests": 100,
                    }
                ],
            }
        },
    }


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(create_validate_router())
    # Surface handler exceptions as 500s rather than re-raising, so a regression
    # is reported as a status-code assertion instead of a test error.
    return TestClient(app, raise_server_exceptions=False)


class TestValidateRouteResponseContract:
    """Every manifest gets a 200 with a structured verdict, never a 500."""

    def test_valid_manifest_passes(self, client: TestClient) -> None:
        response = client.post("/api/v1/validate", json={"manifest": _valid_manifest()})

        assert response.status_code == 200
        assert response.json() == {"passed": True, "errors": [], "warnings": []}

    @pytest.mark.parametrize(
        "divisor",
        [
            param(0, id="int-zero"),
            param(0.0, id="float-zero"),
            param(-0.0, id="negative-float-zero"),
            param(False, id="yaml-false"),
            param(1.0e-320, id="denormal-1e-320"),
            param(5e-324, id="smallest-subnormal"),
            param(-1, id="negative"),
            param("abc", id="unparsable-string"),
            param(None, id="null"),
        ],
    )  # fmt: skip
    def test_hostile_connections_per_worker_returns_structured_errors(
        self, client: TestClient, divisor: object
    ) -> None:
        manifest = _valid_manifest()
        manifest["spec"]["connectionsPerWorker"] = divisor

        response = client.post("/api/v1/validate", json={"manifest": manifest})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["passed"] is False
        assert body["errors"], "a rejected manifest must explain why"
        assert isinstance(body["warnings"], list)

    def test_validator_raising_still_answers_within_the_contract(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The unauthenticated route must not leak a future unguarded raise as a 500."""

        def _boom(*_: object, **__: object) -> None:
            raise ZeroDivisionError("division by zero")

        monkeypatch.setattr(mod, "validate_manifest", _boom)

        response = client.post("/api/v1/validate", json={"manifest": _valid_manifest()})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["passed"] is False
        assert any("ZeroDivisionError" in e for e in body["errors"]), body["errors"]
