# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Result-join barrier eviction of producers that die without reporting.

A producer killed by OOMKill, node eviction or SIGKILL -- the ordinary
Kubernetes deaths -- never emits SERVICE_ERROR. Before eviction existed, such a
service stayed in the barrier's required set forever, ``pending_domains`` never
emptied, and the controller hung after profiling.
"""

from __future__ import annotations

from aiperf.controller.result_join_coordinator import ResultJoinCoordinator

DOMAIN = "records"
OTHER_DOMAIN = "telemetry"


def test_barrier_blocks_while_a_registered_producer_never_reports() -> None:
    """The hang this fixes: a silent producer keeps the barrier pending."""
    coord = ResultJoinCoordinator()
    coord.register(DOMAIN, "records-1")
    coord.register(DOMAIN, "records-2")
    coord.complete(DOMAIN, "records-1")

    assert coord.ready is False
    assert coord.pending_domains == (DOMAIN,)


def test_evicting_a_dead_producer_releases_the_barrier() -> None:
    coord = ResultJoinCoordinator()
    coord.register(DOMAIN, "records-1")
    coord.register(DOMAIN, "records-2")
    coord.complete(DOMAIN, "records-1")

    assert coord.evict_service("records-2", "missed heartbeats") is True
    assert coord.ready is True
    assert coord.pending_domains == ()


def test_eviction_is_recorded_rather_than_silently_satisfying_the_barrier() -> None:
    """A degraded run must be able to name the member that vanished."""
    coord = ResultJoinCoordinator()
    coord.register(DOMAIN, "records-2")

    coord.evict_service("records-2", "pod 'aiperf-w-3' is Failed")

    assert coord.evicted == {"records-2": "pod 'aiperf-w-3' is Failed"}


def test_evicting_an_unknown_service_reports_no_change() -> None:
    """A non-member is not a producer, so callers route its death elsewhere."""
    coord = ResultJoinCoordinator()
    coord.register(DOMAIN, "records-1")

    assert coord.evict_service("not-a-member", "missed heartbeats") is False
    assert coord.evicted == {}
    assert coord.pending_domains == (DOMAIN,)


def test_a_result_arriving_after_eviction_clears_the_degradation() -> None:
    """Guards the opposite failure: dropping a member that was actually alive.

    The barrier has already been released by then, so the late result cannot
    un-release it -- but the run should no longer be reported as degraded on
    that producer's account.
    """
    coord = ResultJoinCoordinator()
    coord.register(DOMAIN, "records-2")
    coord.evict_service("records-2", "missed heartbeats")
    assert coord.evicted

    coord.complete(DOMAIN, "records-2")

    assert coord.evicted == {}
    assert coord.ready is True


def test_eviction_spans_every_domain_the_service_joined() -> None:
    coord = ResultJoinCoordinator()
    coord.register(DOMAIN, "svc-1")
    coord.register(OTHER_DOMAIN, "svc-1")
    coord.register(OTHER_DOMAIN, "svc-2")
    coord.complete(OTHER_DOMAIN, "svc-2")

    coord.evict_service("svc-1", "node evicted")

    assert coord.ready is True


def test_eviction_leaves_other_producers_required() -> None:
    coord = ResultJoinCoordinator()
    coord.register(DOMAIN, "records-1")
    coord.register(DOMAIN, "records-2")

    coord.evict_service("records-2", "missed heartbeats")

    assert coord.ready is False
    assert coord.pending_domains == (DOMAIN,)
    coord.complete(DOMAIN, "records-1")
    assert coord.ready is True


def test_evicting_a_completed_producer_is_not_degraded() -> None:
    """A telemetry pod that delivered results and is then OOMKilled must not
    be reported as a missing producer: its results are already present and
    correct in the export, so eviction must not degrade the run.

    The return stays True because it reports barrier *membership*: the pod is
    still a producer, and the caller must handle it on the producer path.
    Degradation is ``evicted``'s answer, and it is empty here.
    """
    coord = ResultJoinCoordinator()
    coord.register(OTHER_DOMAIN, "tm-1")
    coord.register(DOMAIN, "records-1")
    coord.complete_domain(OTHER_DOMAIN)

    assert coord.evict_service("tm-1", "pod OOMKilled") is True
    assert coord.evicted == {}
    assert coord.pending_domains == (DOMAIN,)


def test_evicting_a_producer_still_pending_in_another_domain_still_degrades() -> None:
    """A producer that completed one domain but is still pending in another
    must still be reported as degraded if it dies before finishing.
    """
    coord = ResultJoinCoordinator()
    coord.register(DOMAIN, "svc-1")
    coord.register(OTHER_DOMAIN, "svc-1")
    coord.complete_domain(DOMAIN)

    assert coord.evict_service("svc-1", "missed heartbeats") is True
    assert coord.evicted == {"svc-1": "missed heartbeats"}
