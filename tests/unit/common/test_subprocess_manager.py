# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SubprocessManager process-supervision primitive."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.subprocess_manager import SubprocessInfo, SubprocessManager
from aiperf.plugin.enums import ServiceType
from tests.unit.common.conftest import make_subprocess_info


@contextmanager
def mock_mp_process(mock_process: MagicMock):
    """Patch get_mp_context so ``context.Process(...)`` returns *mock_process*.

    Yields the Process constructor mock so callers can assert on call kwargs.
    """
    mock_ctx = MagicMock()
    mock_ctx.Process.return_value = mock_process
    with patch(
        "aiperf.common.subprocess_manager.get_mp_context", return_value=mock_ctx
    ):
        yield mock_ctx.Process


class TestSubprocessInfo:
    """Tests for the SubprocessInfo dataclass."""

    def test_create_with_process_stores_all_fields(
        self, mock_process_alive: MagicMock
    ) -> None:
        info = SubprocessInfo(
            process=mock_process_alive,
            service_type=ServiceType.WORKER,
            service_id="worker_001",
        )
        assert info.service_type == ServiceType.WORKER
        assert info.service_id == "worker_001"
        assert info.pid == 12345
        assert info.exitcode is None

    def test_no_process_yields_none_pid_and_exitcode(self) -> None:
        info = make_subprocess_info(process=None)
        assert info.pid is None
        assert info.exitcode is None

    def test_dead_process_exposes_exitcode(
        self, mock_process_crashed: MagicMock
    ) -> None:
        info = make_subprocess_info(process=mock_process_crashed)
        assert info.exitcode == 1


class TestSpawnService:
    """Tests for spawn_service / spawn_services."""

    @pytest.mark.asyncio
    async def test_spawn_service_returns_tracked_info(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        with mock_mp_process(mock_process_alive):
            info = await subprocess_manager.spawn_service(
                ServiceType.WORKER, service_id="worker-0"
            )

        assert info.service_id == "worker-0"
        assert info.service_type == ServiceType.WORKER
        assert subprocess_manager.get_by_type(ServiceType.WORKER) == [info]
        # The local group-manager boundary is started first and tracked too, so
        # the worker is the second (and last) tracked child, not the only one.
        assert subprocess_manager.subprocesses[-1] is info
        assert [p.service_type for p in subprocess_manager.subprocesses] == [
            ServiceType.WORKER_GROUP_MANAGER,
            ServiceType.WORKER,
        ]

    @pytest.mark.asyncio
    async def test_spawn_service_generates_id_when_replicable(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        with mock_mp_process(mock_process_alive):
            info = await subprocess_manager.spawn_service(
                ServiceType.WORKER, replicable=True
            )

        assert info.service_id.startswith(f"{ServiceType.WORKER}_")
        assert info.service_id != str(ServiceType.WORKER)

    @pytest.mark.asyncio
    async def test_spawn_service_uses_service_type_id_when_not_replicable(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        with mock_mp_process(mock_process_alive):
            info = await subprocess_manager.spawn_service(
                ServiceType.TIMING_MANAGER, replicable=False
            )

        assert info.service_id == str(ServiceType.TIMING_MANAGER)

    @pytest.mark.asyncio
    async def test_spawn_service_passes_run_and_queues_to_child(
        self, benchmark_run, mock_process_alive: MagicMock
    ) -> None:
        log_queue = MagicMock()
        error_queue = MagicMock()
        manager = SubprocessManager(
            run=benchmark_run, log_queue=log_queue, error_queue=error_queue
        )

        with mock_mp_process(mock_process_alive) as process_ctor:
            await manager.spawn_service(ServiceType.WORKER, service_id="worker-0")

        kwargs = process_ctor.call_args.kwargs["kwargs"]
        assert kwargs["run"] is benchmark_run
        assert kwargs["log_queue"] is log_queue
        assert kwargs["error_queue"] is error_queue
        assert kwargs["service_id"] == "worker-0"
        assert isinstance(kwargs["controller_pid"], int)

    @pytest.mark.asyncio
    async def test_spawn_service_forwards_none_error_queue_when_unset(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        """``bootstrap_and_run_service`` declares the parameter, so None is safe.

        It must still be passed explicitly rather than omitted: omission would
        silently reintroduce the caller-less error path this replaced.
        """
        with mock_mp_process(mock_process_alive) as process_ctor:
            await subprocess_manager.spawn_service(ServiceType.WORKER)

        assert process_ctor.call_args.kwargs["kwargs"]["error_queue"] is None

    @pytest.mark.asyncio
    async def test_ordinary_service_is_spawned_daemonic(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        with mock_mp_process(mock_process_alive) as process_ctor:
            await subprocess_manager.spawn_service(ServiceType.WORKER)

        assert process_ctor.call_args.kwargs["daemon"] is True

    @pytest.mark.asyncio
    async def test_worker_group_manager_is_spawned_non_daemonic(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        """The WGM forks its own children, which daemonic processes may not do."""
        with mock_mp_process(mock_process_alive) as process_ctor:
            await subprocess_manager.spawn_service(
                ServiceType.WORKER_GROUP_MANAGER, service_id="wgm"
            )

        assert process_ctor.call_args.kwargs["daemon"] is False

    @pytest.mark.asyncio
    async def test_group_manager_boundary_is_started_before_first_child(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        with mock_mp_process(mock_process_alive):
            info = await subprocess_manager.spawn_service(ServiceType.WORKER)

        adapter = subprocess_manager.local_worker_group_runtime_adapter
        assert adapter is not None
        assert info.launch_adapter is adapter
        assert info.parent_service_id == adapter.service_id
        assert len(subprocess_manager.subprocesses) == 2

    @pytest.mark.asyncio
    async def test_sibling_children_share_one_group_manager_adapter(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        """Every child must carry the *running* boundary's adapter instance.

        Regression guard: rebuilding a candidate adapter per spawn and returning
        it handed each sibling its own copy, so they could diverge from the live
        boundary's identity and declared capacity.
        """
        with mock_mp_process(mock_process_alive):
            first = await subprocess_manager.spawn_service(
                ServiceType.WORKER, service_id="worker-0"
            )
            second = await subprocess_manager.spawn_service(
                ServiceType.RECORD_PROCESSOR, service_id="rp-0"
            )

        assert first.launch_adapter is second.launch_adapter
        assert (
            first.launch_adapter
            is subprocess_manager.local_worker_group_runtime_adapter
        )
        # Exactly one boundary, despite two children.
        assert (
            len(subprocess_manager.get_by_type(ServiceType.WORKER_GROUP_MANAGER)) == 1
        )

    @pytest.mark.asyncio
    async def test_spawn_services_creates_requested_replica_count(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        with mock_mp_process(mock_process_alive):
            infos = await subprocess_manager.spawn_services(ServiceType.WORKER, 3)

        assert len(infos) == 3
        assert len({info.service_id for info in infos}) == 3
        assert len(subprocess_manager.get_by_type(ServiceType.WORKER)) == 3


class TestStopAndKill:
    """Tests for stop_process / stop_service / stop_all / kill_all."""

    @pytest.mark.asyncio
    async def test_stop_process_terminates_live_child(
        self, subprocess_manager: SubprocessManager, mock_process_factory
    ) -> None:
        # Alive at entry, dead after terminate() so no kill() is needed.
        process = mock_process_factory(is_alive=True)
        process.is_alive.side_effect = [True, False]
        info = make_subprocess_info(process=process)

        await subprocess_manager.stop_process(info)

        process.terminate.assert_called_once()
        process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_process_kills_child_that_ignores_terminate(
        self, subprocess_manager_with_logger, mock_process_factory
    ) -> None:
        manager, mock_logger = subprocess_manager_with_logger
        process = mock_process_factory(is_alive=True)
        info = make_subprocess_info(process=process, service_id="stubborn")

        await manager.stop_process(info)

        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        assert any(
            "stubborn" in str(call.args[0])
            for call in mock_logger.warning.call_args_list
        )

    @pytest.mark.asyncio
    async def test_stop_process_is_noop_for_dead_child(
        self, subprocess_manager: SubprocessManager, mock_process_dead: MagicMock
    ) -> None:
        await subprocess_manager.stop_process(
            make_subprocess_info(process=mock_process_dead)
        )
        mock_process_dead.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_service_only_stops_matching_type(
        self, subprocess_manager: SubprocessManager, mock_process_factory
    ) -> None:
        worker = make_subprocess_info(
            service_type=ServiceType.WORKER,
            service_id="w0",
            process=mock_process_factory(is_alive=False),
        )
        timing = make_subprocess_info(
            service_type=ServiceType.TIMING_MANAGER,
            service_id="t0",
            process=mock_process_factory(is_alive=False),
        )
        subprocess_manager.subprocesses.extend([worker, timing])

        await subprocess_manager.stop_service(ServiceType.WORKER)

        assert subprocess_manager.subprocesses == [timing]

    @pytest.mark.asyncio
    async def test_stop_all_returns_one_result_per_child(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        with mock_mp_process(mock_process_alive):
            await subprocess_manager.spawn_service(
                ServiceType.WORKER, service_id="worker-0"
            )
            await subprocess_manager.spawn_service(
                ServiceType.WORKER, service_id="worker-1"
            )

        results = await subprocess_manager.stop_all()

        # Two workers plus the one local group-manager boundary they share.
        assert len(results) == 3
        assert subprocess_manager.subprocesses == []

    @pytest.mark.asyncio
    async def test_kill_all_kills_every_live_child_and_clears_tracking(
        self, subprocess_manager: SubprocessManager, mock_process_factory
    ) -> None:
        alive = mock_process_factory(is_alive=True)
        dead = mock_process_factory(is_alive=False)
        subprocess_manager.subprocesses.extend(
            [
                make_subprocess_info(process=alive, service_id="a"),
                make_subprocess_info(process=dead, service_id="d"),
            ]
        )

        results = await subprocess_manager.kill_all()

        alive.kill.assert_called_once()
        dead.kill.assert_not_called()
        assert len(results) == 2
        assert subprocess_manager.subprocesses == []


class TestTrackingHelpers:
    """Tests for get_by_type / check_alive / remove / clear."""

    @pytest.mark.asyncio
    async def test_check_alive_drops_dead_children(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        with mock_mp_process(mock_process_alive):
            info = await subprocess_manager.spawn_service(
                ServiceType.WORKER, service_id="worker-0"
            )

        assert subprocess_manager.check_alive() == []

        mock_process_alive.is_alive.return_value = False
        # Both the worker and its group-manager boundary share the mock process.
        assert info in subprocess_manager.check_alive()

    def test_check_alive_ignores_entries_without_process(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        subprocess_manager.subprocesses.append(make_subprocess_info(process=None))
        assert subprocess_manager.check_alive() == []

    def test_get_by_type_filters_by_service_type(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        worker = make_subprocess_info(service_type=ServiceType.WORKER, service_id="w")
        timing = make_subprocess_info(
            service_type=ServiceType.TIMING_MANAGER, service_id="t"
        )
        subprocess_manager.subprocesses.extend([worker, timing])

        assert subprocess_manager.get_by_type(ServiceType.WORKER) == [worker]

    def test_remove_untracks_without_error_when_absent(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        info = make_subprocess_info()
        subprocess_manager.remove(info)  # not tracked: must not raise
        subprocess_manager.subprocesses.append(info)
        subprocess_manager.remove(info)
        assert subprocess_manager.subprocesses == []

    def test_clear_drops_all_tracking(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        subprocess_manager.subprocesses.append(make_subprocess_info())
        subprocess_manager.clear()
        assert subprocess_manager.subprocesses == []
