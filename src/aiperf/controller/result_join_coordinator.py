# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tracks post-profile result producers until every registered result is joined."""

from __future__ import annotations


class ResultJoinCoordinator:
    """Coordinates readiness across registered result-producing domains."""

    def __init__(self) -> None:
        self._required: dict[str, set[str]] = {}
        self._completed: dict[str, set[str]] = {}
        self._last_reported_pending: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.pending_domains == ()

    @property
    def pending_domains(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                domain
                for domain, required in self._required.items()
                if required - self._completed.get(domain, set())
            )
        )

    def register(self, domain: str, service_id: str) -> None:
        self._required.setdefault(domain, set()).add(service_id)

    def unregister(self, domain: str, service_id: str) -> None:
        required = self._required.get(domain)
        if required is None:
            return

        required.discard(service_id)
        completed = self._completed.get(domain)
        if completed is not None:
            completed.discard(service_id)

        if not required:
            self._required.pop(domain, None)
            self._completed.pop(domain, None)

    def unregister_service(self, service_id: str) -> None:
        for domain in tuple(self._required):
            self.unregister(domain, service_id)

    def complete(self, domain: str, service_id: str) -> None:
        if service_id not in self._required.get(domain, set()):
            return
        self._completed.setdefault(domain, set()).add(service_id)

    def complete_domain(self, domain: str) -> None:
        required = self._required.get(domain)
        if not required:
            return
        self._completed[domain] = set(required)

    def pending_domains_changed(self) -> tuple[str, ...] | None:
        pending = self.pending_domains
        if pending == self._last_reported_pending:
            return None
        self._last_reported_pending = pending
        return pending
