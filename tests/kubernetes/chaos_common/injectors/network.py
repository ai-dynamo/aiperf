# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified :py:class:`FaultInjector` for the ``network.*`` fault domain.

Adapts the existing :py:class:`ToxiproxyInjector` (``tests/kubernetes/chaos/
toxiproxy.py``) to the unified chaos interface defined in
``tests/kubernetes/chaos_common/base.py``. One Toxiproxy toxic type per
``network.*`` fault_id:

* ``network.latency``     -> ``latency`` toxic
* ``network.timeout``     -> ``timeout`` toxic
* ``network.bandwidth``   -> ``bandwidth`` toxic
* ``network.reset_peer``  -> ``reset_peer`` toxic
* ``network.slow_close``  -> ``slow_close`` toxic
* ``network.partition``   -> full proxy disable (``PATCH /proxies/<name>``
  with ``enabled: false``), restored by re-enabling.

Phase 2 scope: toxic add/remove + proxy disable. The Toxiproxy proxy
itself is set up at fixture time (see ``tests/kubernetes/chaos/conftest.py``
``toxiproxy_injector``); this injector only mutates toxics on existing
proxies. Phase 3 will add ``cluster.network_policy.deny_egress`` for the
egress-blackhole case under a separate ``ClusterNetworkInjector``.
"""

from __future__ import annotations

from typing import Any, ClassVar
from uuid import uuid4

import aiohttp
import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos.toxiproxy import ToxiproxyError, ToxiproxyInjector
from tests.kubernetes.chaos_common.base import (
    AppliedFault,
    FaultInjector,
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)

logger = AIPerfLogger(__name__)


PARTITION_FAULT_ID = "network.partition"
"""Sentinel fault_id that disables the whole proxy instead of adding a toxic."""

_FAULT_TO_TOXIC_TYPE: dict[str, str] = {
    "network.latency": "latency",
    "network.timeout": "timeout",
    "network.bandwidth": "bandwidth",
    "network.reset_peer": "reset_peer",
    "network.slow_close": "slow_close",
}
"""Mapping from unified fault_id to Toxiproxy toxic type."""


class _NetworkToxicApplied(AppliedFault):
    """Restore handle for a single Toxiproxy toxic added by :py:class:`NetworkInjector`."""

    def __init__(
        self,
        spec: FaultSpec,
        toxiproxy: ToxiproxyInjector,
        proxy_name: str,
        toxic_name: str,
    ) -> None:
        super().__init__(
            spec=spec,
            metadata={"proxy_name": proxy_name, "toxic_name": toxic_name},
        )
        self._toxiproxy = toxiproxy
        self._proxy_name = proxy_name
        self._toxic_name = toxic_name

    async def restore(self) -> None:
        try:
            await self._toxiproxy.remove_toxic(
                proxy_name=self._proxy_name,
                toxic_name=self._toxic_name,
            )
        except ToxiproxyError as exc:
            # Tolerate "already gone": proxy may have been deleted by a
            # sibling fault or test cleanup. Per base.py cleanup contract
            # (§5), restore is idempotent and swallows benign failures so
            # the original test exception is not masked.
            logger.warning(
                "NetworkInjector: remove_toxic(%s, %s) failed; swallowing: %s",
                self._proxy_name,
                self._toxic_name,
                exc,
            )
        except aiohttp.ClientError as exc:
            raise FaultMechanismError(
                f"NetworkInjector: transport error removing toxic "
                f"{self._toxic_name!r} on proxy {self._proxy_name!r}: {exc}"
            ) from exc


class _NetworkPartitionApplied(AppliedFault):
    """Restore handle that re-enables a Toxiproxy proxy disabled by a partition."""

    def __init__(
        self,
        spec: FaultSpec,
        toxiproxy: ToxiproxyInjector,
        proxy_name: str,
    ) -> None:
        super().__init__(spec=spec, metadata={"proxy_name": proxy_name})
        self._toxiproxy = toxiproxy
        self._proxy_name = proxy_name

    async def restore(self) -> None:
        try:
            await _patch_proxy_enabled(self._toxiproxy, self._proxy_name, enabled=True)
        except ToxiproxyError as exc:
            logger.warning(
                "NetworkInjector: re-enabling proxy %s failed; swallowing: %s",
                self._proxy_name,
                exc,
            )
        except aiohttp.ClientError as exc:
            raise FaultMechanismError(
                f"NetworkInjector: transport error re-enabling proxy "
                f"{self._proxy_name!r}: {exc}"
            ) from exc


async def _patch_proxy_enabled(
    toxiproxy: ToxiproxyInjector,
    proxy_name: str,
    *,
    enabled: bool,
) -> None:
    """PATCH ``/proxies/<name>`` with ``enabled`` flag.

    Implemented locally (instead of extending :py:class:`ToxiproxyInjector`)
    because the legacy class is owned by ``chaos/`` and the unified-chaos
    spec keeps adapter-only logic on this side of the boundary. Follows the
    same short-lived ``aiohttp.ClientSession`` discipline as ``_post_json``
    (see toxiproxy.py module docstring on event-loop scope).
    """
    payload = orjson.dumps({"enabled": enabled})
    url = f"{toxiproxy.base_url}/proxies/{proxy_name}"
    async with (
        aiohttp.ClientSession(timeout=toxiproxy._client_timeout()) as session,
        session.request(
            "POST",
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        ) as resp,
    ):
        body = await resp.read()
        if resp.status >= 400:
            raise ToxiproxyError(
                f"toxiproxy POST /proxies/{proxy_name} -> {resp.status}: "
                f"{body.decode(errors='replace')}"
            )


class NetworkInjector(FaultInjector):
    """Adapter exposing the ``network.*`` fault domain via Toxiproxy.

    The :py:class:`ToxiproxyInjector` is expected to be initialized and
    have the relevant proxy already configured by a test fixture (e.g.
    ``toxiproxy_injector`` in ``tests/kubernetes/chaos/conftest.py``).
    This injector translates a :py:class:`FaultSpec` of the form::

        FaultSpec(
            fault_id="network.latency",
            target={"proxy": "mock-server"},
            params={
                "attributes": {"latency": 500, "jitter": 50},
                "stream": "downstream",       # optional
                "toxicity": 1.0,              # optional
            },
        )

    into a single ``add_toxic`` call and tracks the auto-generated toxic
    name for restore.

    ``fault_id="network.partition"`` disables the proxy via PATCH
    ``/proxies/<name>`` with ``enabled: false``; restore re-enables it.
    """

    HANDLES: ClassVar[tuple[str, ...]] = ("network",)

    def __init__(self, toxiproxy: ToxiproxyInjector) -> None:
        self._toxiproxy = toxiproxy

    async def inject(self, spec: FaultSpec) -> AppliedFault:
        proxy_name = spec.target.get("proxy")
        if not proxy_name:
            raise FaultPreconditionError(
                f"NetworkInjector: spec.target['proxy'] is required for "
                f"fault_id={spec.fault_id!r}; got target={spec.target!r}"
            )

        if spec.fault_id == PARTITION_FAULT_ID:
            return await self._inject_partition(spec, proxy_name)

        toxic_type = _FAULT_TO_TOXIC_TYPE.get(spec.fault_id)
        if toxic_type is None:
            raise FaultPreconditionError(
                f"NetworkInjector: unsupported fault_id={spec.fault_id!r}; "
                f"known: {sorted(_FAULT_TO_TOXIC_TYPE) + [PARTITION_FAULT_ID]}"
            )

        attributes = spec.params.get("attributes")
        if attributes is None:
            raise FaultPreconditionError(
                f"NetworkInjector: spec.params['attributes'] is required for "
                f"fault_id={spec.fault_id!r}; got params={spec.params!r}"
            )

        # Deterministic-but-collision-resistant: <toxic_type>-<6 hex>. The
        # short hex tail keeps log lines readable while avoiding clashes
        # when a test stacks two latency toxics on one proxy.
        toxic_name = f"{toxic_type}-{uuid4().hex[:6]}"

        kwargs: dict[str, Any] = {
            "proxy_name": proxy_name,
            "toxic_type": toxic_type,
            "attributes": attributes,
            "name": toxic_name,
        }
        if "stream" in spec.params:
            kwargs["stream"] = spec.params["stream"]
        if "toxicity" in spec.params:
            kwargs["toxicity"] = spec.params["toxicity"]

        try:
            await self._toxiproxy.add_toxic(**kwargs)
        except ToxiproxyError as exc:
            raise FaultMechanismError(
                f"NetworkInjector: add_toxic({toxic_type!r}) on proxy "
                f"{proxy_name!r} failed: {exc}"
            ) from exc
        except aiohttp.ClientError as exc:
            raise FaultMechanismError(
                f"NetworkInjector: transport error adding {toxic_type!r} "
                f"toxic on proxy {proxy_name!r}: {exc}"
            ) from exc

        return _NetworkToxicApplied(
            spec=spec,
            toxiproxy=self._toxiproxy,
            proxy_name=proxy_name,
            toxic_name=toxic_name,
        )

    async def _inject_partition(
        self,
        spec: FaultSpec,
        proxy_name: str,
    ) -> AppliedFault:
        try:
            await _patch_proxy_enabled(self._toxiproxy, proxy_name, enabled=False)
        except ToxiproxyError as exc:
            raise FaultMechanismError(
                f"NetworkInjector: disabling proxy {proxy_name!r} failed: {exc}"
            ) from exc
        except aiohttp.ClientError as exc:
            raise FaultMechanismError(
                f"NetworkInjector: transport error disabling proxy "
                f"{proxy_name!r}: {exc}"
            ) from exc

        return _NetworkPartitionApplied(
            spec=spec,
            toxiproxy=self._toxiproxy,
            proxy_name=proxy_name,
        )
