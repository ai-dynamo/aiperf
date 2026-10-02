# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified :py:class:`FaultInjector` for the ``client.*`` fault domain.

Pure client-side faults targeting an HTTP server, used to assert server-side
cancellation paths and body-size-limit enforcement end-to-end. No kubectl,
no Toxiproxy, no Kubernetes resources are mutated -- the only state in flight
is an :py:class:`aiohttp.ClientSession` we own.

Supported fault_ids:

* ``client.cancel_request``  -> POST + read stream chunks, then force-close
  the session mid-flight to drop the TCP socket. The k8s-side analog of
  dynamo's ``tests/fault_tolerance/cancellation/utils.py`` ``CancellableRequest``
  pattern, but built on aiohttp + ``session.close()`` instead of a socket
  monkey-patch.
* ``client.overflow_tokens`` -> POST a payload whose oversized field exceeds
  the server's configured body limit (e.g. ``DYN_HTTP_BODY_LIMIT_MB``), then
  capture the rejection response (expected HTTP 413).

Both faults are one-shot mutations of an external service from the client
side; :py:meth:`restore` is a no-op for both because the fault is the
request itself, not a state change to be undone.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import aiohttp

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.base import (
    AppliedFault,
    FaultInjector,
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)

logger = AIPerfLogger(__name__)


CANCEL_FAULT_ID = "client.cancel_request"
"""Fault_id for the mid-stream cancellation dispatch."""

OVERFLOW_FAULT_ID = "client.overflow_tokens"
"""Fault_id for the oversized-prompt body-limit dispatch."""

_BODY_PREVIEW_LIMIT_BYTES = 4 * 1024
"""Maximum bytes of the server response body stashed in ``metadata['body_preview']``."""

_CANCEL_READ_CHUNK_SIZE = 64 * 1024
"""Chunk size used when reading streamed bytes before the cancel deadline."""


class _ClientCancelApplied(AppliedFault):
    """Restore handle for a completed mid-stream cancel; ``restore()`` is a no-op."""

    async def restore(self) -> None:
        # The cancel already happened in inject(); nothing to undo. The base
        # class's _restored guard prevents double-invocation.
        return


class _ClientOverflowApplied(AppliedFault):
    """Restore handle for a completed oversize-payload POST; ``restore()`` is a no-op."""

    async def restore(self) -> None:
        return


class ClientInjector(FaultInjector):
    """In-process client-side fault injector for the ``client.*`` domain.

    No external dependencies: each :py:meth:`inject` call owns its own
    short-lived :py:class:`aiohttp.ClientSession` so the injector can be
    reused across tests without lingering connection state.

    Example::

        spec = FaultSpec(
            fault_id="client.cancel_request",
            target={"url": "http://frontend.dynamo-system.svc:8000/v1/chat/completions"},
            params={
                "payload": {"model": "x", "messages": [...], "stream": True},
                "cancel_after_seconds": 0.25,
            },
        )
        async with await ClientInjector().inject(spec) as handle:
            assert handle.metadata["bytes_received"] >= 0
    """

    HANDLES: ClassVar[tuple[str, ...]] = ("client",)

    def __init__(self) -> None:
        pass

    async def inject(self, spec: FaultSpec) -> AppliedFault:
        url = spec.target.get("url")
        if not url:
            raise FaultPreconditionError(
                f"ClientInjector: spec.target['url'] is required for "
                f"fault_id={spec.fault_id!r}; got target={spec.target!r}"
            )

        if spec.fault_id == CANCEL_FAULT_ID:
            return await self._inject_cancel(spec, url)
        if spec.fault_id == OVERFLOW_FAULT_ID:
            return await self._inject_overflow(spec, url)

        raise FaultPreconditionError(
            f"ClientInjector: unknown client fault_id: {spec.fault_id!r}; "
            f"known: [{CANCEL_FAULT_ID!r}, {OVERFLOW_FAULT_ID!r}]"
        )

    async def _inject_cancel(self, spec: FaultSpec, url: str) -> AppliedFault:
        payload = spec.params.get("payload")
        if payload is None:
            raise FaultPreconditionError(
                f"ClientInjector: spec.params['payload'] is required for "
                f"fault_id={spec.fault_id!r}"
            )
        cancel_after = spec.params.get("cancel_after_seconds")
        if cancel_after is None:
            raise FaultPreconditionError(
                f"ClientInjector: spec.params['cancel_after_seconds'] is required "
                f"for fault_id={spec.fault_id!r}"
            )

        bytes_received = 0
        last_chunk_monotonic: float | None = None
        exception_type: str | None = None

        session = aiohttp.ClientSession()
        try:
            try:
                resp_cm = session.post(url, json=payload)
                resp = await resp_cm.__aenter__()
            except aiohttp.ClientConnectionError as exc:
                raise FaultMechanismError(
                    f"ClientInjector: could not establish connection to {url!r}: {exc}"
                ) from exc
            except aiohttp.ClientError as exc:
                raise FaultMechanismError(
                    f"ClientInjector: could not establish connection to {url!r}: {exc}"
                ) from exc

            async def _drain() -> None:
                nonlocal bytes_received, last_chunk_monotonic
                async for chunk in resp.content.iter_chunked(_CANCEL_READ_CHUNK_SIZE):
                    bytes_received += len(chunk)
                    last_chunk_monotonic = asyncio.get_event_loop().time()

            drain_task = asyncio.create_task(_drain())
            try:
                await asyncio.wait_for(drain_task, timeout=float(cancel_after))
            except TimeoutError:
                # Expected path: deadline hit before server finished streaming.
                # Cancel the read task; we will close the session below to drop
                # the TCP socket, which is the actual fault under test.
                drain_task.cancel()
                try:
                    await drain_task
                except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
                    exception_type = type(exc).__name__
            except asyncio.CancelledError:
                exception_type = "CancelledError"
                raise
            except Exception as exc:  # noqa: BLE001
                exception_type = type(exc).__name__

            # release() returns the connection to the pool without reading the
            # rest of the body; close() then drops the socket itself. Together
            # this is the in-process equivalent of dynamo's
            # CancellableRequest.cancel() socket.shutdown + socket.close.
            resp.release()
        finally:
            await session.close()

        metadata: dict[str, Any] = {
            "url": url,
            "bytes_received": bytes_received,
            "last_chunk_monotonic": last_chunk_monotonic,
            "exception_type": exception_type,
        }
        return _ClientCancelApplied(spec=spec, metadata=metadata)

    async def _inject_overflow(self, spec: FaultSpec, url: str) -> AppliedFault:
        template = spec.params.get("payload_template")
        if template is None:
            raise FaultPreconditionError(
                f"ClientInjector: spec.params['payload_template'] is required "
                f"for fault_id={spec.fault_id!r}"
            )
        size_bytes = spec.params.get("payload_size_bytes")
        if size_bytes is None:
            raise FaultPreconditionError(
                f"ClientInjector: spec.params['payload_size_bytes'] is required "
                f"for fault_id={spec.fault_id!r}"
            )
        prompt_field = spec.params.get("prompt_field", "prompt")

        payload: dict[str, Any] = dict(template)
        payload[prompt_field] = "x" * int(size_bytes)

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(url, json=payload) as resp,
            ):
                raw = await resp.content.read(_BODY_PREVIEW_LIMIT_BYTES)
                metadata: dict[str, Any] = {
                    "url": url,
                    "status": resp.status,
                    "headers": dict(resp.headers),
                    "body_preview": raw.decode(errors="replace"),
                    "payload_size_bytes": int(size_bytes),
                    "prompt_field": prompt_field,
                }
        except aiohttp.ClientError as exc:
            raise FaultMechanismError(
                f"ClientInjector: network failure POSTing oversized payload "
                f"to {url!r}: {exc}"
            ) from exc

        return _ClientOverflowApplied(spec=spec, metadata=metadata)
