# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI-based AIPerf API Service.

Provides HTTP endpoints for metrics and status, plus WebSocket streaming
for real-time ZMQ message forwarding.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Annotated

import uvicorn
from fastapi import Depends, FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import HTTPConnection
from starlette_compress import CompressMiddleware

from aiperf import __version__ as aiperf_version
from aiperf.api.routers.base_router import BaseRouter
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_start, on_stop
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceType

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def get_service(conn: HTTPConnection) -> FastAPIService:
    """Get FastAPIService from app state. Works for both HTTP and WebSocket."""
    service = getattr(conn.app.state, "service", None)
    if service is None:
        raise RuntimeError("Service not initialized in app.state")
    return service


ServiceDep = Annotated["FastAPIService", Depends(get_service)]


class FastAPIService(BaseComponentService):
    """FastAPI-based API Service.

    Provides HTTP endpoints for metrics and status, plus WebSocket streaming
    for real-time ZMQ message forwarding.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )

        self.api_host = service_config.api_host or Environment.API_SERVER.HOST
        self.api_port = service_config.api_port or Environment.API_SERVER.PORT
        self.cors_origins = Environment.API_SERVER.CORS_ORIGINS

        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None
        self._stop_task: asyncio.Task | None = None

        self._routers: dict[str, BaseRouter] = {}
        self._load_routers()

        self.app = self._create_app()

    def _load_routers(self) -> None:
        """Instantiate BaseRouter plugins and attach as child lifecycles."""
        for entry in plugins.iter_entries(PluginType.API_ROUTER):
            cls = entry.load()
            router = cls(
                service_config=self.service_config,
                user_config=self.user_config,
            )
            self._routers[entry.name] = router
            self.attach_child_lifecycle(router)

    @property
    def _base_url(self) -> str:
        """Get the base URL for the API server."""
        return f"http://{self.api_host}:{self.api_port}"

    def _create_app(self) -> FastAPI:
        """Create the FastAPI application with all routes."""
        service = self

        @asynccontextmanager
        async def lifespan(_: FastAPI) -> AsyncIterator[None]:
            service.info(f"FastAPI starting at {service._base_url}/")
            yield
            service.info("FastAPI stopped")

        app = FastAPI(
            title="AIPerf API",
            description="Real-time benchmark metrics and WebSocket streaming",
            version=aiperf_version,
            lifespan=lifespan,
        )

        app.add_middleware(
            CompressMiddleware,
            zstd_level=Environment.COMPRESSION.ZSTD_LEVEL,
            gzip_level=Environment.COMPRESSION.GZIP_LEVEL,
        )

        if service.cors_origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=service.cors_origins,
                allow_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
                allow_headers=["*"],
            )

        # Store service in app.state for dependency injection (health, config endpoints)
        app.state.service = service

        # Store routers in app.state keyed by plugin registry name, and include them in the app
        for name, router in self._routers.items():
            setattr(app.state, name, router)
            app.include_router(router.get_router())

        return app

    @on_start
    async def _start_api_server(self) -> None:
        """Start the FastAPI server."""
        if self.api_port is None:
            raise ValueError(
                "API port is not configured. Set --api-port or AIPERF_API_SERVER_PORT."
            )
        config = uvicorn.Config(
            self.app,
            host=self.api_host,
            port=self.api_port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        self._server_task.add_done_callback(self._on_server_task_done)

        self.info(f"AIPerf FastAPI started at {self._base_url}/")
        self.info(
            lambda: "  Routes: "
            + " | ".join(
                r.path
                for r in self.app.routes
                if hasattr(r, "methods") and r.path not in ("/openapi.json",)
            )
        )

    def _on_server_task_done(self, task: asyncio.Task[None]) -> None:
        """Surface unhandled server errors and trigger graceful shutdown."""
        if task.cancelled():
            return
        if exc := task.exception():
            self.exception(f"FastAPI server failed: {exc!r}")
            self._stop_task = asyncio.get_running_loop().create_task(self.stop())

    @on_stop
    async def _stop_api_server(self) -> None:
        """Stop the FastAPI server."""
        self.info("Stopping AIPerf FastAPI server...")

        if self._server:
            self._server.should_exit = True
        if self._server_task:
            try:
                await asyncio.wait_for(
                    self._server_task,
                    timeout=Environment.API_SERVER.SHUTDOWN_TIMEOUT,
                )
            except asyncio.TimeoutError:
                self._server_task.cancel()
                with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._server_task,
                        timeout=Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
                    )
            except asyncio.CancelledError:
                raise

        self.info("AIPerf FastAPI server stopped")


def main() -> None:
    """Main entry point."""
    bootstrap_and_run_service(ServiceType.API)


if __name__ == "__main__":
    main()
