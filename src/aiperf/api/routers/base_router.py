# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for lifecycle-managed router components."""

from __future__ import annotations

from abc import abstractmethod

from fastapi import APIRouter, Depends
from starlette.requests import HTTPConnection

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin


class BaseRouter(AIPerfLifecycleMixin):
    """Base class for lifecycle-managed router components.

    Subclasses that need message bus access get it through their composed mixin
    (e.g. RealtimeMetricsMixin -> MessageBusClientMixin -> CommunicationMixin).
    The base class itself only requires lifecycle management.
    """

    def __init__(
        self,
        user_config: UserConfig,
        service_config: ServiceConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.user_config = user_config
        self.service_config = service_config

    @abstractmethod
    def get_router(self) -> APIRouter:
        """Return the APIRouter for this component."""
        ...


def component_dependency(state_attr: str) -> Depends:
    """Create a FastAPI ``Depends`` that resolves a component from ``app.state``."""

    def _resolve(conn: HTTPConnection) -> BaseRouter:
        return getattr(conn.app.state, state_attr)

    return Depends(_resolve)
