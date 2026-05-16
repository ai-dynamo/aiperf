# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for lifecycle-managed router components."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from starlette.requests import HTTPConnection

from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class BaseRouter(AIPerfLifecycleMixin):
    """Base class for lifecycle-managed router components.

    Subclasses that need message bus access get it through their composed mixin
    (e.g. RealtimeMetricsMixin -> MessageBusClientMixin -> CommunicationMixin).
    The base class itself only requires lifecycle management.
    """

    def __init__(
        self,
        run: BenchmarkRun,
        **kwargs,
    ) -> None:
        super().__init__(run=run, **kwargs)

    @abstractmethod
    def get_router(self) -> APIRouter:
        """Return the APIRouter for this component."""
        ...


def component_dependency(state_attr: str) -> Depends:
    """Create a FastAPI ``Depends`` that resolves a component from ``app.state``."""

    def _resolve(conn: HTTPConnection) -> BaseRouter:
        return getattr(conn.app.state, state_attr)

    return Depends(_resolve)
