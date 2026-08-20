# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI-based AIPerf API Service.

Provides HTTP endpoints for metrics and status, plus WebSocket streaming
for real-time ZMQ message forwarding.
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette_compress import CompressMiddleware

from aiperf import __version__ as aiperf_version
from aiperf.api.depends import ServiceDep, get_service
from aiperf.api.routers.base_router import BaseRouter
from aiperf.api.routers.tokenizer import build_tokenizer_router
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.enums import CommandType
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_command, on_start, on_stop
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceRunType, ServiceType

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aiperf.common.messages import CommandMessage
    from aiperf.config import BenchmarkRun


# Re-exported from `aiperf.api.depends` so existing imports of
# `get_service` / `ServiceDep` from `aiperf.api.api_service` keep working.
__all__ = ["FastAPIService", "ServiceDep", "get_service", "main"]


class FastAPIService(BaseComponentService):
    """FastAPI-based API Service.

    Provides HTTP endpoints for metrics and status, plus WebSocket streaming
    for real-time ZMQ message forwarding.
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )

        self.api_host = run.cfg.runtime.api_host or Environment.API_SERVER.HOST
        self.api_port = (
            self._api_port or run.cfg.runtime.api_port or Environment.API_SERVER.PORT
        )
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
                run=self.run,
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

        # Mount the tokenizer-bundle router (plain APIRouter factory, not a
        # BaseRouter plugin: it has no lifecycle and only closes over the
        # registry).
        app.include_router(build_tokenizer_router())

        return app

    @on_start
    async def _start_api_server(self) -> None:
        """Start the FastAPI server."""
        if self.api_port is None:
            raise ValueError(
                "API port is not configured. Set --api-port or AIPERF_API_SERVER_PORT."
            )
        # Pre-bind probe: catch port conflicts BEFORE uvicorn schedules the
        # async serve() task. Without this, bind failure surfaces inside an
        # asyncio task done-callback after credits have already drained, and
        # the run silently "succeeds" with no API. There is a TOCTOU race
        # vs uvicorn's actual bind, but it's tight enough that "port already
        # bound" failures are caught reliably for the user-explicit case.
        explicit_port = self.run.cfg.runtime.api_port is not None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((self.api_host, self.api_port))
        except OSError as e:
            msg = f"API server cannot bind {self.api_host}:{self.api_port}: {e}"
            if explicit_port:
                # User-explicit --api-port; surface as fatal so the controller
                # aborts via process-monitor → pod_failure_abort_event.
                raise RuntimeError(msg) from e
            self.warning(f"{msg}; continuing without API server.")
            return
        # Pre-warm the shared HF cache before binding the port. Worker pods
        # hit `/api/tokenizer/{name}/bundle` as soon as the WGM comes up;
        # without this, they 503-retry while dataset-manager incidentally
        # populates the cache via its own `_configure_tokenizer` load. With
        # this, the bundle endpoint serves on the first attempt.
        await self._prewarm_tokenizers()

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

    def _tokenizers_to_warm(self) -> list[str]:
        """Tokenizer names the api container should pre-fetch into the shared HF cache.

        Mirrors ``WorkerGroupManager._unique_tokenizer_names``: explicit
        ``cfg.tokenizer.name`` wins; otherwise fall back to model names.

        Excludes the local-only names (``builtin`` and the tiktoken
        encodings) — they are constructed in-process and never travel
        through the bundle endpoint.
        """
        from aiperf.common.tokenizer import (
            BUILTIN_TOKENIZER_NAME,
            TIKTOKEN_ENCODING_NAMES,
        )

        cfg = self.run.cfg
        seen: dict[str, None] = {}
        tokenizer_cfg = getattr(cfg, "tokenizer", None)
        if tokenizer_cfg is not None and getattr(tokenizer_cfg, "name", None):
            seen.setdefault(tokenizer_cfg.name, None)
        else:
            for model_name in cfg.get_model_names():
                seen.setdefault(model_name, None)
        return [
            n
            for n in seen
            if n != BUILTIN_TOKENIZER_NAME and n not in TIKTOKEN_ENCODING_NAMES
        ]

    async def _prewarm_tokenizers(self) -> None:
        """Populate the shared HF cache for every configured tokenizer.

        Runs before uvicorn binds the port so the bundle endpoint never
        returns 503 due to a cold cache. Uses ``AutoTokenizer.from_pretrained``
        (rather than ``snapshot_download`` with a glob) so HuggingFace's
        own logic picks the minimal file set — avoids pulling unrelated
        siblings like ``onnx/`` that would only inflate the bundle and the
        wire bytes shipped to every worker pod.

        After ``from_pretrained`` populates ``HF_HOME``, calls
        :func:`tokenizer_router.prewarm_bundle` to tar+zstd the snapshot
        directory into the router's module-level RAM cache. This guarantees
        the bundle endpoint serves the first request synchronously out of
        memory -- the materialisation cost is paid here once at startup,
        not on the request path where slow tar+compression would manifest
        as worker-pod download timeouts (the request times out client-side
        before the server finishes, the server-side task gets cancelled,
        the cache stays empty, and every retry pays the same cost again).

        Failures (HF egress error, bundle materialisation error) are logged
        but not raised: the bundle endpoint will rebuild on demand and
        surface a clear 503/404 if the snapshot is genuinely missing.
        """
        from transformers import AutoTokenizer

        from aiperf.api.routers.tokenizer import prewarm_bundle

        names = self._tokenizers_to_warm()
        if not names:
            return
        cfg = self.run.cfg
        tokenizer_cfg = getattr(cfg, "tokenizer", None)
        trust_remote_code = bool(
            getattr(tokenizer_cfg, "trust_remote_code", False)
            if tokenizer_cfg is not None
            else False
        )
        revision = (
            getattr(tokenizer_cfg, "revision", "main")
            if tokenizer_cfg is not None
            else "main"
        )
        self.info(f"Pre-warming tokenizers into shared HF cache: {names}")

        async def _warm_one(name: str) -> None:
            try:
                # Loaded into RAM once for cache-population side effect, then
                # GC'd. AutoTokenizer downloads only the files HF knows the
                # tokenizer needs (~5 files for gpt2; weights/onnx skipped).
                await asyncio.to_thread(
                    AutoTokenizer.from_pretrained,
                    name,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                )
            except Exception as exc:  # noqa: BLE001
                self.warning(
                    f"Pre-warm of tokenizer '{name}' failed ({exc!r}); "
                    f"bundle endpoint will retry on first request"
                )
                return
            try:
                # Tar+zstd into the router's module-level cache so the
                # bundle endpoint hits memory, not disk-+-CPU, on every
                # request. This is the fix for worker pods timing out on
                # first /api/tokenizer/{name}/bundle (b/c tar+zstd of a
                # large snapshot dir was happening synchronously inside
                # the request, exceeding the 300s aiohttp client timeout
                # and never completing because each retry restarted the
                # work from scratch).
                await prewarm_bundle(name)
                self.info(f"Pre-warmed tokenizer '{name}'")
            except Exception as exc:  # noqa: BLE001
                self.warning(
                    f"Pre-warm bundle materialisation for '{name}' failed "
                    f"({exc!r}); bundle endpoint will rebuild on first request"
                )

        await asyncio.gather(*(_warm_one(n) for n in names))

    @on_command(CommandType.SHUTDOWN)
    async def _on_shutdown_command(self, message: CommandMessage) -> None:
        """Ignore the controller's broadcast shutdown under Kubernetes.

        In Kubernetes the controller pod deliberately outlives its benchmark so
        `aiperf kube results` can read from it, and is retired explicitly via
        POST /api/shutdown -- what `aiperf kube shutdown` and the operator's
        graceful-exit handshake drive. Honouring the broadcast races that
        design: the listener goes away a few seconds after the run ends, which
        is shorter than the operator's monitor interval, so the operator loses
        the endpoint between two polls and the AIPerfJob never leaves its
        pre-terminal phase. Every other run type still stops on the broadcast,
        keeping ``Environment.API_SERVER.POST_COMPLETE_GRACE`` (607977c1a7,
        DYN-701) exactly as it behaves today -- the two mechanisms solve the
        same problem and must not both run.
        """
        if self.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES:
            self.info(
                "Kubernetes mode: ignoring broadcast shutdown; the API stays up "
                "to serve results until POST /api/shutdown arrives."
            )
            return
        await super()._on_shutdown_command(message)

    @on_stop
    async def _stop_api_server(self) -> None:
        """Stop the FastAPI server."""
        # Keep the listener open for a grace window so clients polling /api/results
        # can observe terminal status before connection-refused. See
        # Environment.API_SERVER.POST_COMPLETE_GRACE. Skip when there's no live
        # serve task (startup failure, server crashed, or already finished) — a
        # closed listener can't be kept open.
        grace = Environment.API_SERVER.POST_COMPLETE_GRACE
        server_running = self._server_task is not None and not self._server_task.done()
        if grace > 0 and server_running:
            self.info(
                f"Holding API listener open for {grace:.1f}s "
                "to let polling clients observe terminal status."
            )
            await asyncio.sleep(grace)

        self.info("Stopping AIPerf FastAPI server...")

        if self._server:
            self._server.should_exit = True
        if self._server_task:
            try:
                await asyncio.wait_for(
                    self._server_task,
                    timeout=Environment.API_SERVER.SHUTDOWN_TIMEOUT,
                )
            except TimeoutError:
                self._server_task.cancel()
                with suppress(asyncio.CancelledError, TimeoutError):
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
