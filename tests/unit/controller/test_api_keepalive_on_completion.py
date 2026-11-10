# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The API must survive the controller long enough to serve final results.

Two defects compounded here. The completion signal once had no publisher, so
/api/results never left "running" and POST /api/shutdown answered 409 forever
-- and that endpoint is the operator's graceful-exit handshake, so every
completion fell through to a hard pod delete. Separately, the grace window that
keeps the API listener alive was gated on _is_api_service_alive(), which is
unconditionally False under Kubernetes because KubernetesServiceManager
inherits an always-empty multi_process_info from MultiProcessServiceManager.

The signal is now a BENCHMARK_COMPLETE command the controller awaits, so it can
neither go unsent nor be lost silently (NVBugs 7073234).
"""

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pytest import param

from aiperf.common.control_structs import (
    CommandAck,
    CommandErr,
    CommandOk,
    CommandUnhandled,
)
from aiperf.common.enums import CommandType, LifecycleState, ServiceRegistrationStatus
from aiperf.common.environment import Environment
from aiperf.common.models import ErrorDetails
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceRunType, ServiceType

API_ID = "api"


def _controller(*, mp_info, api_registered=True):
    ctrl = SystemController.__new__(SystemController)
    ctrl.service_manager = MagicMock()
    api = MagicMock()
    api.registration_status = (
        ServiceRegistrationStatus.REGISTERED
        if api_registered
        else ServiceRegistrationStatus.UNREGISTERED
    )
    api.state = LifecycleState.RUNNING
    ctrl.service_manager.service_map = {ServiceType.API: [api]}
    ctrl.service_manager.multi_process_info = mp_info
    return ctrl


def _ok(*, has_final_results: bool) -> CommandOk:
    return CommandOk(
        cid="c",
        cmd=CommandType.BENCHMARK_COMPLETE,
        sid=API_ID,
        payload=orjson.dumps({"has_final_results": has_final_results}),
    )


def _announcer(
    *,
    api_enabled: bool = True,
    was_cancelled: bool = False,
    reply=None,
    kubernetes: bool = False,
):
    """A controller whose control channel answers BENCHMARK_COMPLETE with ``reply``."""
    ctrl = SystemController.__new__(SystemController)
    ctrl._api_enabled = api_enabled
    ctrl._was_cancelled = was_cancelled
    ctrl.service_id = "system_controller"
    ctrl.run = MagicMock()
    ctrl.run.cfg.runtime.service_run_type = (
        ServiceRunType.KUBERNETES if kubernetes else ServiceRunType.MULTIPROCESSING
    )
    ctrl.warning = MagicMock()
    ctrl.info = MagicMock()
    ctrl._send_control_command_to_all = AsyncMock(
        return_value=[reply if reply is not None else _ok(has_final_results=True)]
    )
    return ctrl


class TestApiLivenessUnderKubernetes:
    def test_registered_api_counts_as_alive_with_no_local_processes(self):
        """K8s runs the API as its own pod; there is no local process to poll."""
        ctrl = _controller(mp_info=[])
        assert ctrl._is_api_service_alive() is True

    def test_unregistered_api_is_not_alive(self):
        ctrl = _controller(mp_info=[], api_registered=False)
        assert ctrl._is_api_service_alive() is False

    def test_local_process_check_still_applies_when_present(self):
        """Multiprocess mode keeps the authoritative is_alive() cross-check."""
        rec = MagicMock()
        rec.service_type = ServiceType.API
        rec.process = MagicMock(is_alive=lambda: False)
        ctrl = _controller(mp_info=[rec])
        assert ctrl._is_api_service_alive() is False

    def test_live_local_process_is_alive(self):
        rec = MagicMock()
        rec.service_type = ServiceType.API
        rec.process = MagicMock(is_alive=lambda: True)
        ctrl = _controller(mp_info=[rec])
        assert ctrl._is_api_service_alive() is True


class TestBenchmarkCompleteHandshake:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("was_cancelled", "kubernetes", "settled"),
        [
            param(False, False, True, id="completed-local"),
            param(True, False, True, id="cancelled-local"),
            param(False, True, False, id="completed-kubernetes"),
        ],
    )  # fmt: skip
    async def test_terminal_state_is_sent_to_the_api_and_awaited(
        self,
        was_cancelled: bool,
        kubernetes: bool,
        settled: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without this the results endpoint never completes and shutdown 409s.

        ``results_settled`` is true only where the controller has joined
        RecordsManager before the handshake (local); under Kubernetes its final
        results may still be in flight.
        """
        monkeypatch.setattr(Environment.API_SERVER, "COMPLETE_ACK_TIMEOUT", 3.0)
        ctrl = _announcer(was_cancelled=was_cancelled, kubernetes=kubernetes)

        await ctrl._announce_benchmark_complete({API_ID})

        ctrl._send_control_command_to_all.assert_awaited_once_with(
            CommandType.BENCHMARK_COMPLETE,
            [API_ID],
            payload=orjson.dumps(
                {"was_cancelled": was_cancelled, "results_settled": settled}
            ),
            timeout=3.0,
        )
        ctrl.warning.assert_not_called()
        ctrl.info.assert_called_once_with(
            "API 'api' confirmed the run's terminal state"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("api_enabled", "api_ids"),
        [
            param(False, {API_ID}, id="api-disabled"),
            param(True, set(), id="api-never-registered"),
        ],
    )  # fmt: skip
    async def test_nothing_is_sent_without_a_registered_api(
        self, api_enabled: bool, api_ids: set[str]
    ) -> None:
        ctrl = _announcer(api_enabled=api_enabled)

        await ctrl._announce_benchmark_complete(api_ids)

        ctrl._send_control_command_to_all.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("was_cancelled", "kubernetes", "expected"),
        [
            param(False, False, "will report 'incomplete' now", id="completed-local"),
            param(True, False, "will report 'cancelled' now", id="cancelled-local"),
            param(True, True, "will report 'cancelled' once its wait for late final results lapses", id="cancelled-kubernetes"),
        ],
    )  # fmt: skip
    async def test_missing_final_results_warns_with_the_status_clients_will_see(
        self, was_cancelled: bool, kubernetes: bool, expected: str
    ) -> None:
        """The warning must describe what a client polling within the hold sees."""
        ctrl = _announcer(
            was_cancelled=was_cancelled,
            reply=_ok(has_final_results=False),
            kubernetes=kubernetes,
        )

        await ctrl._announce_benchmark_complete({API_ID})

        ctrl.warning.assert_called_once()
        message = ctrl.warning.call_args.args[0]
        assert "not the final results" in message
        assert message.endswith(expected)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reply", "detail"),
        [
            param(
                ErrorDetails(type="TimeoutError", message="Command benchmark_complete timed out for api"),
                "timed out",
                id="timeout",
            ),
            param(
                CommandErr(cid="c", cmd=CommandType.BENCHMARK_COMPLETE, sid=API_ID, error="router exploded"),
                "router exploded",
                id="handler-error",
            ),
            param(
                CommandUnhandled(cid="c", cmd=CommandType.BENCHMARK_COMPLETE, sid=API_ID),
                "CommandUnhandled",
                id="no-handler",
            ),
            param(
                CommandAck(cid="c", cmd=CommandType.BENCHMARK_COMPLETE, sid=API_ID),
                "CommandAck",
                id="ack-without-result",
            ),
        ],
    )  # fmt: skip
    async def test_unconfirmed_terminal_state_warns_and_does_not_raise(
        self, reply, detail: str
    ) -> None:
        """Shutdown must proceed whatever the API answers."""
        ctrl = _announcer(reply=reply)

        await ctrl._announce_benchmark_complete({API_ID})

        ctrl.warning.assert_called_once()
        message = ctrl.warning.call_args.args[0]
        assert "did not confirm the terminal state" in message
        assert detail in message
