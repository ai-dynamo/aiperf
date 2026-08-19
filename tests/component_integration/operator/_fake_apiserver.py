# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal in-memory fake of kubernetes_asyncio's CustomObjectsApi.

Component-integration tests for operator handlers patch
``aiperf.kubernetes.client.k8s_client`` and ``kubernetes_asyncio.client.CustomObjectsApi``
so that get/patch round-trips hit this in-memory store rather than a real
apiserver. The store records every call so tests can assert call counts,
patch shapes, and ordering.

Public surface:
    FakeApiserver(crs)          - holds CRs keyed by (ns, plural, name)
    fake.add_cr(...)            - register a CR body
    fake.context()              - patches both k8s_client + CustomObjectsApi
    fake.get_call_count(name)   - count of get_namespaced_custom_object calls
    fake.patch_call_count(name) - count of patch_namespaced_custom_object calls
    fake.patches                - recorded (key, body) tuples in order
    fake.patch_404              - set of keys for which patch raises 404
"""

from __future__ import annotations

import contextlib
import copy
import importlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from kubernetes_asyncio.client.exceptions import ApiException


class FakeApiserver:
    """In-memory fake apiserver shared across one test scenario."""

    def __init__(self) -> None:
        self.crs: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.gets: list[tuple[str, str, str]] = []
        self.patches: list[
            tuple[tuple[str, str, str], dict[str, Any] | list[dict[str, Any]]]
        ] = []
        self.patch_404: set[tuple[str, str, str]] = set()

    def add_cr(
        self, namespace: str, plural: str, name: str, body: dict[str, Any]
    ) -> None:
        self.crs[(namespace, plural, name)] = body

    def get_call_count(self, name: str) -> int:
        return sum(1 for k in self.gets if k[2] == name)

    def patch_call_count(self, name: str) -> int:
        return sum(1 for k, _ in self.patches if k[2] == name)

    async def _get(
        self, *, namespace: str, plural: str, name: str, **_: Any
    ) -> dict[str, Any]:
        key = (namespace, plural, name)
        self.gets.append(key)
        body = self.crs.get(key)
        if body is None:
            raise ApiException(status=404, reason="Not Found")
        return copy.deepcopy(body)

    async def _patch(
        self,
        *,
        namespace: str,
        plural: str,
        name: str,
        body: dict[str, Any] | list[dict[str, Any]],
        **_: Any,
    ) -> dict[str, Any]:
        key = (namespace, plural, name)
        self.patches.append((key, copy.deepcopy(body)))
        if key in self.patch_404:
            raise ApiException(status=404, reason="Not Found")
        existing = self.crs.get(key)
        if existing is None:
            raise ApiException(status=404, reason="Not Found")
        if isinstance(body, dict):
            _merge(existing, body)
        # JSON-patch op bodies are recorded; tests assert on fake.patches
        # directly without mutating the stored CR.
        return copy.deepcopy(existing)

    @contextlib.contextmanager
    def context(self) -> Any:
        """Patch ``k8s_client`` (yields a no-op context) and ``CustomObjectsApi``.

        ``k8s_client`` must be patched at every module-level binding site, not
        just its home module: operator modules that do
        ``from aiperf.kubernetes.client import k8s_client`` at import time hold
        their own reference, and patching only ``aiperf.kubernetes.client``
        leaves those bindings pointing at the real client — which then loads
        the developer's actual ``~/.kube/config`` (function-local imports
        resolve through the patched home module at call time and need no
        entry here).
        """

        class _ApiCtx:
            async def __aenter__(_self) -> Any:
                return object()

            async def __aexit__(_self, *_a: Any) -> None:
                return None

        custom = MagicMock()
        custom.get_namespaced_custom_object = AsyncMock(side_effect=self._get)
        custom.patch_namespaced_custom_object = AsyncMock(side_effect=self._patch)
        k8s_client_binding_sites = [
            "aiperf.kubernetes.client.k8s_client",
            "aiperf.operator.client_cache.k8s_client",
            "aiperf.operator.handlers.lifecycle.k8s_client",
            "aiperf.operator.handlers.create.k8s_client",
            "aiperf.operator.handlers.completion.k8s_client",
        ]
        with contextlib.ExitStack() as stack:
            for site in k8s_client_binding_sites:
                module_name, attribute = site.rsplit(".", maxsplit=1)
                module = importlib.import_module(module_name)
                if hasattr(module, attribute):
                    stack.enter_context(patch(site, new=lambda: _ApiCtx()))
            stack.enter_context(
                patch("kubernetes_asyncio.client.CustomObjectsApi", return_value=custom)
            )
            yield self


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
