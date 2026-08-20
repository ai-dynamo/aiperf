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
        self._evicted: dict[str, str] = {}

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

    def evict_service(self, service_id: str, reason: str) -> bool:
        """Drop a dead producer from the barrier and record why.

        A service that dies abruptly -- OOMKilled, evicted, SIGKILLed, i.e. the
        common Kubernetes death -- never emits an error message, so waiting for
        one means the barrier never releases and the controller hangs after
        profiling. Liveness failure is the authoritative eviction signal, as in
        Spark (TaskSetManager.executorLost synthesizes failures for in-flight
        tasks and decrements the success count so the stage can finish) and the
        Kubernetes Job controller (deleted pods are counted as failed "to
        account for orphan Pods that never have a chance to reach the Failed
        phase").

        The eviction is recorded rather than silently satisfying the barrier, so
        the run can be reported as degraded and name the member that vanished.

        Returns True if the service was actually required (i.e. the barrier
        changed), so callers can skip a redundant readiness re-check.
        """
        was_required = any(
            service_id in required for required in self._required.values()
        )
        self.unregister_service(service_id)
        if was_required:
            self._evicted[service_id] = reason
        return was_required

    @property
    def evicted(self) -> dict[str, str]:
        """Services dropped from the barrier because they died, by reason."""
        return dict(self._evicted)

    def complete(self, domain: str, service_id: str) -> None:
        # A result arriving after eviction is accepted, not discarded: the
        # member turned out to have produced its output, so the run is no
        # longer degraded on its account. The barrier has already been
        # released, so this only clears the degradation record.
        if self._evicted.pop(service_id, None) is not None:
            return
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
