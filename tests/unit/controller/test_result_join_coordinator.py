# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.controller.result_join_coordinator import ResultJoinCoordinator


def test_ready_when_no_result_producers_registered() -> None:
    coord = ResultJoinCoordinator()

    assert coord.ready
    assert coord.pending_domains == ()


def test_register_marks_domain_pending_until_service_completes() -> None:
    coord = ResultJoinCoordinator()

    coord.register("telemetry", "t1")

    assert not coord.ready
    assert coord.pending_domains == ("telemetry",)

    coord.complete("telemetry", "t1")

    assert coord.ready
    assert coord.pending_domains == ()


def test_multiple_services_in_same_domain_all_must_complete() -> None:
    coord = ResultJoinCoordinator()

    coord.register("telemetry", "t1")
    coord.register("telemetry", "t2")
    coord.complete("telemetry", "t1")

    assert not coord.ready
    assert coord.pending_domains == ("telemetry",)

    coord.complete("telemetry", "t2")

    assert coord.ready
    assert coord.pending_domains == ()


def test_complete_domain_marks_all_participants_complete() -> None:
    coord = ResultJoinCoordinator()

    coord.register("telemetry", "t1")
    coord.register("telemetry", "t2")
    coord.complete_domain("telemetry")

    assert coord.ready
    assert coord.pending_domains == ()


def test_unregister_removes_pending_participant() -> None:
    coord = ResultJoinCoordinator()

    coord.register("telemetry", "t1")
    coord.unregister("telemetry", "t1")

    assert coord.ready
    assert coord.pending_domains == ()


def test_unregister_service_removes_service_from_all_domains() -> None:
    coord = ResultJoinCoordinator()

    coord.register("telemetry", "t1")
    coord.register("server_metrics", "t1")
    coord.register("profile", "records")

    coord.unregister_service("t1")

    assert coord.pending_domains == ("profile",)


def test_complete_unknown_participant_does_not_create_domain() -> None:
    coord = ResultJoinCoordinator()

    coord.complete("telemetry", "t1")

    assert coord.ready
    assert coord.pending_domains == ()


def test_complete_unknown_domain_does_not_create_domain() -> None:
    coord = ResultJoinCoordinator()

    coord.complete_domain("telemetry")

    assert coord.ready
    assert coord.pending_domains == ()


def test_completed_participant_reregistration_stays_complete() -> None:
    coord = ResultJoinCoordinator()

    coord.register("telemetry", "t1")
    coord.complete("telemetry", "t1")
    coord.register("telemetry", "t1")

    assert coord.ready
    assert coord.pending_domains == ()


def test_pending_domains_changed_only_reports_changes() -> None:
    coord = ResultJoinCoordinator()

    assert coord.pending_domains_changed() is None

    coord.register("server_metrics", "s1")
    assert coord.pending_domains_changed() == ("server_metrics",)
    assert coord.pending_domains_changed() is None

    coord.register("telemetry", "t1")
    assert coord.pending_domains_changed() == ("server_metrics", "telemetry")

    coord.complete("telemetry", "t1")
    assert coord.pending_domains_changed() == ("server_metrics",)

    coord.complete("server_metrics", "s1")
    assert coord.pending_domains_changed() == ()
    assert coord.pending_domains_changed() is None
