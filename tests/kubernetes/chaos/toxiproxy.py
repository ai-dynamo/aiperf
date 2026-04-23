# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Toxiproxy client for chaos tests.

Wraps the toxiproxy REST API (:8474) with an async, intent-revealing
interface. Used by `test_chaos_api_disruption.py` to pause the apiserver
and block operator-to-controller HTTP mid-run.

Usage::

    async def test_apiserver_pause(
        toxiproxy_injector: ToxiproxyInjector,
    ) -> None:
        await toxiproxy_injector.add_proxy(
            name="apiserver",
            listen="0.0.0.0:20000",
            upstream="kubernetes.default.svc:443",
        )
        await toxiproxy_injector.add_toxic(
            "apiserver", "timeout", {"timeout": 0}
        )
        # ... observe operator behavior ...
        await toxiproxy_injector.reset()
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

import aiohttp
import orjson

from tests.kubernetes.helpers.kubectl import KubectlClient

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 5.0
"""Default per-request timeout against the toxiproxy admin API."""

TOXIPROXY_NAMESPACE = "aiperf-chaos-toxiproxy"
"""Namespace the fixture manifest deploys into."""

TOXIPROXY_SERVICE = "toxiproxy"
"""Service name fronting the toxiproxy Deployment."""

TOXIPROXY_ADMIN_PORT = 8474
"""Toxiproxy admin REST API port (inside the cluster)."""

TOXIPROXY_MOCK_SERVER_PORT = 20010
"""Reserved listen port for fronting ``aiperf-mock-server`` traffic.

Kept in sync with the named ``mock-server`` port in
``fixtures/toxiproxy.yaml``. Tests that need to inject faults into
benchmark traffic should ``add_proxy(name="mock-server",
listen=f"0.0.0.0:{TOXIPROXY_MOCK_SERVER_PORT}", upstream=...)`` and
point ``AIPerfJobConfig.endpoint_url`` at
``http://toxiproxy.aiperf-chaos-toxiproxy.svc.cluster.local:20010/v1``.
Generic proxy slots (20000-20005) remain unreserved."""


class ToxiproxyInjector:
    """Async REST client for a cluster-deployed toxiproxy instance.

    Intended lifecycle (session-scoped fixture):

    1. ``ensure_deployed(kubectl)`` — apply the fixture manifest and wait
       for the pod to become Ready.
    2. One-or-more proxy/toxic calls per test.
    3. ``reset()`` between tests to wipe state.
    4. ``teardown(kubectl)`` at session end to delete the namespace.

    All methods raise ``aiohttp.ClientError`` subclasses on network errors
    and ``ToxiproxyError`` on unexpected API responses; tests should wrap
    calls in ``try/finally`` + ``reset()`` for hermetic cleanup.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the injector.

        Args:
            base_url: Admin API URL (e.g. ``http://127.0.0.1:8474``). When
                ``None``, ``ensure_deployed`` must be called first to open
                a port-forward and set the URL.
            timeout: Per-request timeout in seconds.
        """
        self._base_url = base_url
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None
        self._pf_stack: AsyncExitStack | None = None

    @property
    def base_url(self) -> str:
        """Return the admin URL; raises RuntimeError if not initialized."""
        if self._base_url is None:
            raise RuntimeError(
                "ToxiproxyInjector base_url is None; call ensure_deployed() first"
            )
        return self._base_url

    async def ensure_deployed(self, kubectl: KubectlClient) -> None:
        """Apply the fixture manifest, wait for Ready, and port-forward admin.

        Idempotent: safe to call multiple times within the same session.
        Opens a port-forward from a local ephemeral port to the toxiproxy
        admin service and sets ``self._base_url`` accordingly.
        """
        from pathlib import Path

        manifest_path = Path(__file__).parent / "fixtures" / "toxiproxy.yaml"
        manifest = manifest_path.read_text()
        await kubectl.apply(manifest)
        ok = await kubectl.wait_for_rollout(
            "deployment",
            TOXIPROXY_SERVICE,
            namespace=TOXIPROXY_NAMESPACE,
            timeout=60,
        )
        if not ok:
            logs = await kubectl.get_logs(
                f"deployment/{TOXIPROXY_SERVICE}",
                namespace=TOXIPROXY_NAMESPACE,
            )
            raise RuntimeError(f"toxiproxy rollout failed; logs:\n{logs}")

        pod_res = await kubectl.run(
            "get",
            "pods",
            "-n",
            TOXIPROXY_NAMESPACE,
            "-l",
            f"app={TOXIPROXY_SERVICE}",
            "-o",
            "jsonpath={.items[0].metadata.name}",
            check=True,
        )
        pod = pod_res.stdout.strip()
        if not pod:
            raise RuntimeError("toxiproxy pod not found after rollout")

        self._pf_stack = AsyncExitStack()
        local_port = await self._pf_stack.enter_async_context(
            kubectl.port_forward(
                pod, TOXIPROXY_ADMIN_PORT, namespace=TOXIPROXY_NAMESPACE
            )
        )
        self._base_url = f"http://127.0.0.1:{local_port}"
        self._session = aiohttp.ClientSession(timeout=self._timeout)

        # Confirm admin API reachable before returning.
        for attempt in range(10):
            try:
                async with self._session.get(f"{self._base_url}/version") as resp:
                    if resp.status == 200:
                        logger.info(
                            "toxiproxy reachable at %s (version=%s)",
                            self._base_url,
                            (await resp.text()).strip(),
                        )
                        return
            except aiohttp.ClientError as exc:
                logger.debug("toxiproxy not ready (attempt %d): %s", attempt, exc)
            await asyncio.sleep(0.5)
        raise RuntimeError(
            f"toxiproxy admin API did not respond at {self._base_url}/version "
            f"after {10 * 0.5} s; check `kubectl get pods -n {TOXIPROXY_NAMESPACE}`"
        )

    async def add_proxy(self, name: str, listen: str, upstream: str) -> dict[str, Any]:
        """Create a new proxy that fronts ``upstream`` on ``listen``.

        Args:
            name: Proxy name (unique per toxiproxy instance).
            listen: Listen address inside the toxiproxy pod, e.g.
                ``"0.0.0.0:20000"``. Must be one of the ports exposed by
                the Service (20000-20005 by default).
            upstream: Upstream target, e.g.
                ``"kubernetes.default.svc:443"``.

        Returns:
            The decoded toxiproxy proxy object.
        """
        payload = {
            "name": name,
            "listen": listen,
            "upstream": upstream,
            "enabled": True,
        }
        return await self._post_json("/proxies", payload)

    async def add_toxic(
        self,
        proxy_name: str,
        toxic_type: str,
        attributes: dict[str, Any],
        *,
        name: str | None = None,
        stream: str = "downstream",
        toxicity: float = 1.0,
    ) -> dict[str, Any]:
        """Attach a toxic to an existing proxy.

        Args:
            proxy_name: Target proxy (must have been created via ``add_proxy``).
            toxic_type: Toxiproxy toxic identifier, e.g. ``"latency"``,
                ``"timeout"``, ``"bandwidth"``, ``"slow_close"``, ``"reset_peer"``.
            attributes: Toxic-specific attributes, e.g. ``{"latency": 1000,
                "jitter": 100}`` for ``latency``.
            name: Optional explicit toxic name (defaults to
                ``"<toxic_type>_<stream>"``).
            stream: ``"upstream"`` or ``"downstream"``.
            toxicity: Fraction of traffic affected (0.0 - 1.0).

        Returns:
            The decoded toxic object.
        """
        payload: dict[str, Any] = {
            "type": toxic_type,
            "stream": stream,
            "toxicity": toxicity,
            "attributes": attributes,
        }
        if name is not None:
            payload["name"] = name
        return await self._post_json(f"/proxies/{proxy_name}/toxics", payload)

    async def remove_toxic(self, proxy_name: str, toxic_name: str) -> None:
        """Delete a toxic by name."""
        await self._delete(f"/proxies/{proxy_name}/toxics/{toxic_name}")

    async def remove_proxy(self, name: str) -> None:
        """Delete a proxy by name."""
        await self._delete(f"/proxies/{name}")

    async def reset(self) -> None:
        """Wipe every proxy and toxic (``POST /reset`` equivalent).

        Implemented by listing proxies and deleting each, which also
        removes attached toxics. ``/reset`` re-enables proxies rather than
        deleting them, so we use DELETE explicitly.
        """
        if self._session is None:
            return
        try:
            async with self._session.get(f"{self.base_url}/proxies") as resp:
                if resp.status != 200:
                    return
                body = await resp.read()
        except aiohttp.ClientError as exc:
            logger.warning("toxiproxy reset: list failed: %s", exc)
            return
        try:
            proxies = orjson.loads(body)
        except orjson.JSONDecodeError:
            return
        for proxy_name in proxies:
            try:
                await self._delete(f"/proxies/{proxy_name}")
            except aiohttp.ClientError as exc:
                logger.warning(
                    "toxiproxy reset: failed to delete proxy %s: %s",
                    proxy_name,
                    exc,
                )

    async def teardown(self, kubectl: KubectlClient) -> None:
        """Delete the toxiproxy namespace and close the port-forward."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._pf_stack is not None:
            await self._pf_stack.aclose()
            self._pf_stack = None
        self._base_url = None
        await kubectl.delete_namespace(TOXIPROXY_NAMESPACE, wait=False)

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to the admin API and return the decoded response body."""
        if self._session is None:
            raise RuntimeError(
                "ToxiproxyInjector session not open; call ensure_deployed"
            )
        async with self._session.post(
            f"{self.base_url}{path}",
            data=orjson.dumps(payload),
            headers={"Content-Type": "application/json"},
        ) as resp:
            body = await resp.read()
            if resp.status >= 400:
                raise ToxiproxyError(
                    f"toxiproxy POST {path} -> {resp.status}: {body.decode(errors='replace')}"
                )
            if not body:
                return {}
            return orjson.loads(body)

    async def _delete(self, path: str) -> None:
        """DELETE a toxiproxy resource; tolerates 404 on already-gone paths."""
        if self._session is None:
            raise RuntimeError(
                "ToxiproxyInjector session not open; call ensure_deployed"
            )
        async with self._session.delete(f"{self.base_url}{path}") as resp:
            if resp.status not in (200, 204, 404):
                body = await resp.read()
                raise ToxiproxyError(
                    f"toxiproxy DELETE {path} -> {resp.status}: {body.decode(errors='replace')}"
                )


class ToxiproxyError(RuntimeError):
    """Raised when the toxiproxy admin API returns an unexpected status.

    The message names the HTTP method, path, status code, and truncated
    response body so test failures identify the offending call without
    needing to re-run under verbose logging.
    """
