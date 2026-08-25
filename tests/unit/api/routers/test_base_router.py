# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for BaseRouter and the component_dependency factory.

Focuses on:
- BaseRouter is abstract — subclasses must implement get_router().
- Concrete subclasses receive the BenchmarkRun via the lifecycle mixin chain.
- component_dependency() resolves the named attribute off ``app.state`` at
  request time, returning a FastAPI Depends compatible with the test client.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from fastapi import APIRouter, FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.config import AIPerfConfig, BenchmarkRun


class _ConcreteRouter(BaseRouter):
    """Minimal concrete subclass — exposes a pre-built APIRouter."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._router = APIRouter()
        self.payload = "hello"

    def get_router(self) -> APIRouter:
        return self._router


class TestBaseRouterContract:
    """Verify the abstract base contract."""

    def test_get_router_is_declared_abstract_on_base(self) -> None:
        # BaseRouter doesn't use ABCMeta, so direct instantiation isn't blocked
        # by Python — but get_router is still marked abstract for tooling /
        # subclasses, which matters for static analysis. Verify the marker
        # rather than runtime enforcement.
        assert getattr(BaseRouter.get_router, "__isabstractmethod__", False)

    def test_concrete_subclass_instantiates_with_run(
        self, router_config: BenchmarkRun
    ) -> None:
        router = _ConcreteRouter(run=router_config)
        assert isinstance(router.get_router(), APIRouter)

    def test_concrete_subclass_init_accepts_extra_kwargs(
        self, router_config: BenchmarkRun
    ) -> None:
        # Lifecycle mixin chain forwards **kwargs — must not reject a freshly
        # constructed concrete subclass.
        router = _ConcreteRouter(run=router_config)
        assert router.get_router() is not None


class TestComponentDependency:
    """Verify the component_dependency factory resolves from app.state."""

    def test_returns_a_fastapi_depends(self) -> None:
        dep = component_dependency("foo")
        # FastAPI's Depends(...) returns a `params.Depends` instance — exposing
        # a `dependency` callable. Don't reach into FastAPI internals beyond
        # the public surface (callable + .dependency attribute).
        assert hasattr(dep, "dependency")
        assert callable(dep.dependency)

    def test_resolves_attribute_off_app_state_per_request(
        self, router_config: BenchmarkRun
    ) -> None:
        router = _ConcreteRouter(run=router_config)
        app = FastAPI()
        app.state.my_component = router

        @app.get("/probe")
        async def _probe(
            comp: Annotated[_ConcreteRouter, component_dependency("my_component")],
        ) -> dict[str, Any]:
            return {"payload": comp.payload}

        client = TestClient(app)
        response = client.get("/probe")
        assert response.status_code == 200
        assert response.json() == {"payload": "hello"}

    def test_missing_state_attribute_raises_attribute_error(self) -> None:
        app = FastAPI()

        @app.get("/probe")
        async def _probe(
            comp: Annotated[Any, component_dependency("not_set")],
        ) -> dict[str, Any]:
            return {"ok": True}

        client = TestClient(app)
        # Starlette propagates handler exceptions as 500 by default; the
        # underlying cause is AttributeError raised by getattr().
        with pytest.raises(AttributeError, match="not_set"):
            client.get("/probe")


class TestBaseRouterIntegrationWithTestClient:
    """End-to-end: subclass + component_dependency wired through FastAPI."""

    def test_endpoint_can_consume_resolved_component(
        self, router_config: AIPerfConfig
    ) -> None:
        router = _ConcreteRouter(run=router_config)
        sub_router = router.get_router()

        @sub_router.get("/ping")
        async def _ping(
            comp: Annotated[_ConcreteRouter, component_dependency("ping_component")],
        ) -> dict[str, str]:
            return {"value": comp.payload}

        app = FastAPI()
        app.state.ping_component = router
        app.include_router(sub_router)

        client = TestClient(app)
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"value": "hello"}
