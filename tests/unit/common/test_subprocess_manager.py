# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SubprocessManager utility class.

Tests cover:
- SubprocessInfo model validation
- Subprocess spawning (single and batch)
- Subprocess stopping (graceful and forced)
- Health monitoring (alive checks)
- Service filtering and removal
- Logger integration
"""

from contextlib import contextmanager
from multiprocessing import Process
from unittest.mock import MagicMock, patch

import pytest
from pytest import param

from aiperf.common.subprocess_manager import SubprocessInfo, SubprocessManager
from aiperf.plugin.enums import ServiceRunType, ServiceType

# Import helper from conftest
from tests.unit.common.conftest import make_subprocess_info


@contextmanager
def mock_mp_process(mock_process: MagicMock):
    """Patch get_mp_context so context.Process(...) returns *mock_process*.

    Yields the Process mock (the constructor), so callers can assert on
    call args the same way as before.
    """
    mock_ctx = MagicMock()
    mock_ctx.Process.return_value = mock_process
    with patch(
        "aiperf.common.subprocess_manager.get_mp_context", return_value=mock_ctx
    ):
        yield mock_ctx.Process


# =============================================================================
# SubprocessInfo Model Tests
# =============================================================================


class TestSubprocessInfo:
    """Tests for the SubprocessInfo dataclass."""

    def test_create_with_process_stores_all_fields(
        self, mock_process_alive: MagicMock
    ) -> None:
        """SubprocessInfo stores process, service_type, and service_id."""
        info = SubprocessInfo(
            process=mock_process_alive,
            service_type=ServiceType.WORKER,
            service_id="worker_001",
        )

        assert info.process == mock_process_alive
        assert info.service_type == ServiceType.WORKER
        assert info.service_id == "worker_001"

    def test_create_with_none_process_for_failed_spawn(self) -> None:
        """SubprocessInfo accepts None process for spawn failures."""
        info = make_subprocess_info(
            service_type=ServiceType.DATASET_MANAGER,
            service_id="dataset_manager",
            process=None,
        )

        assert info.process is None
        assert info.service_type == ServiceType.DATASET_MANAGER

    @pytest.mark.parametrize(
        ("service_type", "service_id"),
        [
            param(ServiceType.WORKER, "worker_0", id="worker"),
            param(ServiceType.RECORD_PROCESSOR, "rp_abc123", id="record_processor"),
            param(ServiceType.TIMING_MANAGER, "timing_manager", id="timing_manager"),
            param(ServiceType.DATASET_MANAGER, "dataset_manager", id="dataset_manager"),
            param(
                ServiceType.WORKER_GROUP_MANAGER,
                "pod_xyz_wgm",
                id="worker_group_manager",
            ),
            param(ServiceType.WORKER_MANAGER, "wm_001", id="worker_manager"),
            param(ServiceType.RECORDS_MANAGER, "rm_001", id="records_manager"),
        ],
    )  # fmt: skip
    def test_create_with_all_service_types(
        self, service_type: ServiceType, service_id: str
    ) -> None:
        """SubprocessInfo works with all supported service types."""
        info = make_subprocess_info(service_type=service_type, service_id=service_id)

        assert info.service_type == service_type
        assert info.service_id == service_id

    def test_default_process_is_none(self) -> None:
        """SubprocessInfo defaults process to None when not provided."""
        info = SubprocessInfo(
            service_type=ServiceType.WORKER,
            service_id="test_worker",
        )

        assert info.process is None
        assert info.service_type == ServiceType.WORKER
        assert info.service_id == "test_worker"


# =============================================================================
# SubprocessManager Initialization Tests
# =============================================================================


class TestSubprocessManagerInit:
    """Tests for SubprocessManager initialization."""

    def test_init_stores_config_and_creates_empty_subprocesses(self, config) -> None:
        """SubprocessManager stores config and initializes empty subprocess list."""
        manager = SubprocessManager(run=config)

        assert manager.run == config
        assert manager.log_queue is None
        assert manager._logger is None
        assert manager.subprocesses == []

    def test_init_accepts_log_queue(self, config) -> None:
        """SubprocessManager stores provided log queue."""
        mock_queue = MagicMock()
        manager = SubprocessManager(
            run=config,
            log_queue=mock_queue,
        )

        assert manager.log_queue == mock_queue

    def test_init_accepts_logger(self, config) -> None:
        """SubprocessManager stores provided logger."""
        mock_logger = MagicMock()
        manager = SubprocessManager(
            run=config,
            logger=mock_logger,
        )

        assert manager._logger == mock_logger


# =============================================================================
# Subprocess Spawning Tests
# =============================================================================


class TestSubprocessManagerSpawn:
    """Tests for subprocess spawning functionality."""

    @pytest.mark.asyncio
    async def test_spawn_service_creates_and_starts_process(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """spawn_service creates Process, starts it, and tracks it."""
        mock_process = MagicMock(spec=Process)
        mock_process.pid = 12345

        with mock_mp_process(mock_process) as MockProcess:
            info = await subprocess_manager.spawn_service(
                service_type=ServiceType.DATASET_MANAGER,
                service_id="dataset_manager",
            )

            MockProcess.assert_called_once()
            mock_process.start.assert_called_once()
            assert info.service_type == ServiceType.DATASET_MANAGER
            assert info.service_id == "dataset_manager"
            assert info in subprocess_manager.subprocesses
            assert info.parent_service_id is None

    @pytest.mark.asyncio
    async def test_spawn_service_generates_unique_id_for_replicable(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """spawn_service auto-generates ID with uuid suffix for replicable services."""
        mock_process = MagicMock(spec=Process)
        mock_process.pid = 11111

        with mock_mp_process(mock_process):
            info = await subprocess_manager.spawn_service(
                service_type=ServiceType.WORKER,
                service_id=None,
                replicable=True,
            )

            assert info.service_id.startswith("worker_")
            assert len(info.service_id) == len("worker_") + 8  # uuid hex suffix

    @pytest.mark.asyncio
    async def test_spawn_service_uses_service_type_string_for_singleton(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """spawn_service uses service type string as ID for non-replicable services."""
        mock_process = MagicMock(spec=Process)
        mock_process.pid = 22222

        with mock_mp_process(mock_process):
            info = await subprocess_manager.spawn_service(
                service_type=ServiceType.DATASET_MANAGER,
                service_id=None,
                replicable=False,
            )

            assert info.service_id == str(ServiceType.DATASET_MANAGER)

    @pytest.mark.asyncio
    async def test_spawn_service_passes_bootstrap_kwargs(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """spawn_service configures Process with correct bootstrap arguments."""
        mock_process = MagicMock(spec=Process)
        mock_process.pid = 33333

        with mock_mp_process(mock_process) as MockProcess:
            await subprocess_manager.spawn_service(
                service_type=ServiceType.DATASET_MANAGER,
                service_id="dataset_manager",
            )

            call_kwargs = MockProcess.call_args.kwargs
            assert call_kwargs["name"] == "dataset_manager_process"
            assert call_kwargs["daemon"] is True

            bootstrap_kwargs = call_kwargs["kwargs"]
            assert bootstrap_kwargs["service_type"] == ServiceType.DATASET_MANAGER
            assert bootstrap_kwargs["service_id"] == "dataset_manager"
            assert bootstrap_kwargs["run"] == subprocess_manager.run

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "num_replicas",
        [
            param(1, id="single"),
            param(3, id="few"),
            param(10, id="many"),
        ],
    )  # fmt: skip
    async def test_spawn_services_creates_specified_replicas(
        self, subprocess_manager: SubprocessManager, num_replicas: int
    ) -> None:
        """spawn_services creates exact number of replicas with unique IDs."""
        mock_process = MagicMock(spec=Process)
        mock_process.pid = 44444

        with mock_mp_process(mock_process) as MockProcess:
            infos = await subprocess_manager.spawn_services(
                service_type=ServiceType.WORKER,
                num_replicas=num_replicas,
            )

            assert len(infos) == num_replicas
            assert MockProcess.call_count == num_replicas + 1
            child_processes = [
                info
                for info in subprocess_manager.subprocesses
                if info.process is not None
            ]
            assert len(child_processes) == num_replicas + 1
            assert (
                subprocess_manager.subprocesses[0].service_id
                == "worker_group_manager_local"
            )

            # All IDs must be unique
            service_ids = [info.service_id for info in infos]
            assert len(set(service_ids)) == num_replicas

    @pytest.mark.asyncio
    async def test_spawn_service_logs_debug_with_pid_and_id(
        self, subprocess_manager_with_logger: tuple[SubprocessManager, MagicMock]
    ) -> None:
        """spawn_service logs debug message containing PID and service ID."""
        manager, mock_logger = subprocess_manager_with_logger
        mock_process = MagicMock(spec=Process)
        mock_process.pid = 55555

        with mock_mp_process(mock_process):
            await manager.spawn_service(
                service_type=ServiceType.WORKER,
                service_id="logged_worker",
            )

            mock_logger.debug.assert_called()
            call_arg = mock_logger.debug.call_args[0][0]
            assert "55555" in call_arg
            assert "logged_worker" in call_arg

    @pytest.mark.asyncio
    async def test_local_mode_starts_worker_group_manager_before_children(
        self, subprocess_manager: SubprocessManager, mock_process_factory
    ) -> None:
        """Local worker launches start a real group-manager boundary before children."""
        subprocess_manager.run.cfg.runtime.service_run_type = (
            ServiceRunType.MULTIPROCESSING
        )
        subprocess_manager.run.cfg.runtime.workers = 4
        subprocess_manager.run.cfg.runtime.record_processors = 2
        group_process = mock_process_factory(pid=60001)
        worker_process = mock_process_factory(pid=60002)
        mock_ctx = MagicMock()
        mock_ctx.Process.side_effect = [group_process, worker_process]

        with patch(
            "aiperf.common.subprocess_manager.get_mp_context", return_value=mock_ctx
        ):
            worker_info = await subprocess_manager.spawn_service(
                service_type=ServiceType.WORKER,
                service_id="worker_0",
            )

        boundary_info = subprocess_manager.subprocesses[0]
        assert boundary_info.service_type == ServiceType.WORKER_GROUP_MANAGER
        assert boundary_info.service_id == "worker_group_manager_local"
        assert boundary_info.process == group_process
        assert worker_info.process == worker_process
        assert worker_info.parent_service_id == boundary_info.service_id
        assert subprocess_manager.subprocesses[1] == worker_info

    @pytest.mark.asyncio
    async def test_local_mode_worker_and_record_processor_capacity_comes_from_adapter(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """Local group-manager adapter exposes both worker and record-processor capacity."""
        subprocess_manager.run.cfg.runtime.service_run_type = (
            ServiceRunType.MULTIPROCESSING
        )
        subprocess_manager.run.cfg.runtime.workers = 5
        subprocess_manager.run.cfg.runtime.record_processors = 3
        mock_process = MagicMock(spec=Process)
        mock_process.pid = 60002

        with mock_mp_process(mock_process):
            await subprocess_manager.spawn_service(
                service_type=ServiceType.WORKER,
                service_id="worker_0",
            )
            await subprocess_manager.spawn_service(
                service_type=ServiceType.RECORD_PROCESSOR,
                service_id="record_processor_0",
            )

        boundary_info = subprocess_manager.subprocesses[0]
        assert boundary_info.launch_adapter is not None
        assert boundary_info.launch_adapter.declared_worker_capacity == 5
        assert boundary_info.launch_adapter.declared_record_processor_capacity == 3
        assert [
            info.service_type
            for info in subprocess_manager.subprocesses
            if info.service_type == ServiceType.WORKER_GROUP_MANAGER
        ] == [ServiceType.WORKER_GROUP_MANAGER]

    @pytest.mark.asyncio
    async def test_local_mode_launch_adapter_is_passed_to_worker_group_manager_runtime(
        self, subprocess_manager: SubprocessManager, mock_process_factory
    ) -> None:
        """Local runtime launches should pass the adapter contract into WorkerGroupManager."""
        subprocess_manager.run.cfg.runtime.service_run_type = (
            ServiceRunType.MULTIPROCESSING
        )
        subprocess_manager.run.cfg.runtime.workers = 5
        subprocess_manager.run.cfg.runtime.record_processors = 3
        group_process = mock_process_factory(pid=60003)
        worker_process = mock_process_factory(pid=60004)
        mock_ctx = MagicMock()
        mock_ctx.Process.side_effect = [group_process, worker_process]

        with patch(
            "aiperf.common.subprocess_manager.get_mp_context", return_value=mock_ctx
        ):
            worker_info = await subprocess_manager.spawn_service(
                service_type=ServiceType.WORKER,
                service_id="worker_0",
            )

        assert worker_info.launch_adapter is not None
        group_launch_kwargs = mock_ctx.Process.call_args_list[0].kwargs["kwargs"]
        runtime_adapter = group_launch_kwargs["runtime_adapter"]
        registration = runtime_adapter.build_registration()

        assert runtime_adapter == subprocess_manager.local_worker_group_runtime_adapter
        assert registration.group_id == "worker_group_manager_local"
        assert registration.declared_workers == 5
        assert registration.declared_record_processors == 3


# =============================================================================
# Subprocess Stopping Tests
# =============================================================================


class TestSubprocessManagerStop:
    """Tests for subprocess stopping functionality."""

    @pytest.mark.asyncio
    async def test_stop_process_terminates_alive_process(
        self, subprocess_manager: SubprocessManager, mock_process_alive: MagicMock
    ) -> None:
        """stop_process calls terminate() on alive process."""
        info = make_subprocess_info(process=mock_process_alive, service_id="to_stop")

        await subprocess_manager.stop_process(info, timeout=1.0)

        mock_process_alive.terminate.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "process_state",
        [
            param("dead", id="dead_process"),
            param("none", id="none_process"),
        ],
    )  # fmt: skip
    async def test_stop_process_skips_non_alive_process(
        self,
        subprocess_manager: SubprocessManager,
        mock_process_dead: MagicMock,
        process_state: str,
    ) -> None:
        """stop_process does not call terminate on dead or None processes."""
        process = mock_process_dead if process_state == "dead" else None
        info = make_subprocess_info(process=process, service_id="skipped")

        await subprocess_manager.stop_process(info, timeout=1.0)

        if process:
            process.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_service_stops_only_matching_type(
        self, subprocess_manager: SubprocessManager, mock_process_factory
    ) -> None:
        """stop_service terminates only processes of specified type and removes them."""
        mock_worker1 = mock_process_factory(pid=1001)
        mock_worker2 = mock_process_factory(pid=1002)
        mock_rp = mock_process_factory(pid=2001)

        rp_info = make_subprocess_info(ServiceType.RECORD_PROCESSOR, "rp1", mock_rp)
        subprocess_manager.subprocesses = [
            make_subprocess_info(ServiceType.WORKER, "w1", mock_worker1),
            make_subprocess_info(ServiceType.WORKER, "w2", mock_worker2),
            rp_info,
        ]

        await subprocess_manager.stop_service(ServiceType.WORKER)

        mock_worker1.terminate.assert_called_once()
        mock_worker2.terminate.assert_called_once()
        mock_rp.terminate.assert_not_called()
        assert subprocess_manager.subprocesses == [rp_info]

    @pytest.mark.asyncio
    async def test_stop_service_stops_only_matching_id(
        self, subprocess_manager: SubprocessManager, mock_process_factory
    ) -> None:
        """stop_service with service_id terminates only that specific process."""
        mock_worker1 = mock_process_factory(pid=1001)
        mock_worker2 = mock_process_factory(pid=1002)

        w2_info = make_subprocess_info(ServiceType.WORKER, "w2", mock_worker2)
        subprocess_manager.subprocesses = [
            make_subprocess_info(ServiceType.WORKER, "w1", mock_worker1),
            w2_info,
        ]

        await subprocess_manager.stop_service(ServiceType.WORKER, service_id="w1")

        mock_worker1.terminate.assert_called_once()
        mock_worker2.terminate.assert_not_called()
        assert subprocess_manager.subprocesses == [w2_info]

    @pytest.mark.asyncio
    async def test_stop_all_terminates_all_subprocesses_and_clears_list(
        self, subprocess_manager: SubprocessManager, mock_process_factory
    ) -> None:
        """stop_all terminates every tracked subprocess and clears the list."""
        mock_procs = [mock_process_factory(pid=3000 + i) for i in range(4)]

        subprocess_manager.subprocesses = [
            make_subprocess_info(ServiceType.WORKER, "w1", mock_procs[0]),
            make_subprocess_info(ServiceType.WORKER, "w2", mock_procs[1]),
            make_subprocess_info(ServiceType.RECORD_PROCESSOR, "rp1", mock_procs[2]),
            make_subprocess_info(ServiceType.TIMING_MANAGER, "tm", mock_procs[3]),
        ]

        await subprocess_manager.stop_all()

        for mock in mock_procs:
            mock.terminate.assert_called_once()

        assert subprocess_manager.subprocesses == []


# =============================================================================
# Subprocess Kill Tests
# =============================================================================


class TestSubprocessManagerKill:
    """Tests for subprocess killing functionality."""

    @pytest.mark.asyncio
    async def test_kill_all_kills_all_processes_immediately(
        self, subprocess_manager: SubprocessManager, mock_process_factory
    ) -> None:
        """kill_all calls kill() on all tracked processes and clears list."""
        mock_procs = [mock_process_factory(pid=4000 + i) for i in range(3)]

        subprocess_manager.subprocesses = [
            make_subprocess_info(ServiceType.WORKER, "w1", mock_procs[0]),
            make_subprocess_info(ServiceType.WORKER, "w2", mock_procs[1]),
            make_subprocess_info(ServiceType.RECORD_PROCESSOR, "rp1", mock_procs[2]),
        ]

        await subprocess_manager.kill_all()

        for mock_proc in mock_procs:
            mock_proc.kill.assert_called_once()
            mock_proc.join.assert_called_once()

        assert subprocess_manager.subprocesses == []


# =============================================================================
# Health Monitoring Tests
# =============================================================================


class TestSubprocessManagerMonitoring:
    """Tests for subprocess health monitoring."""

    def test_check_alive_returns_dead_and_crashed_processes(
        self,
        subprocess_manager: SubprocessManager,
        mock_process_alive: MagicMock,
        mock_process_dead: MagicMock,
        mock_process_crashed: MagicMock,
    ) -> None:
        """check_alive returns processes where is_alive() is False."""
        subprocess_manager.subprocesses = [
            make_subprocess_info(ServiceType.WORKER, "alive", mock_process_alive),
            make_subprocess_info(ServiceType.WORKER, "dead", mock_process_dead),
            make_subprocess_info(ServiceType.WORKER, "crashed", mock_process_crashed),
        ]

        dead = subprocess_manager.check_alive()

        assert len(dead) == 2
        dead_ids = {info.service_id for info in dead}
        assert dead_ids == {"dead", "crashed"}

    def test_check_alive_returns_empty_when_all_alive(
        self, subprocess_manager: SubprocessManager, mock_process_factory
    ) -> None:
        """check_alive returns empty list when all processes are alive."""
        subprocess_manager.subprocesses = [
            make_subprocess_info(ServiceType.WORKER, "w1", mock_process_factory()),
            make_subprocess_info(ServiceType.WORKER, "w2", mock_process_factory()),
        ]

        dead = subprocess_manager.check_alive()

        assert dead == []

    def test_check_alive_ignores_none_processes(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """check_alive does not report None processes as dead."""
        subprocess_manager.subprocesses = [
            make_subprocess_info(ServiceType.WORKER, "never_started", None),
        ]

        dead = subprocess_manager.check_alive()

        assert len(dead) == 0


# =============================================================================
# Service Filtering Tests
# =============================================================================


class TestSubprocessManagerFiltering:
    """Tests for subprocess filtering and retrieval."""

    def test_get_by_type_returns_all_matching(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """get_by_type returns all subprocesses of specified type."""
        subprocess_manager.subprocesses = [
            make_subprocess_info(ServiceType.WORKER, "w1"),
            make_subprocess_info(ServiceType.WORKER, "w2"),
            make_subprocess_info(ServiceType.WORKER, "w3"),
            make_subprocess_info(ServiceType.RECORD_PROCESSOR, "rp1"),
            make_subprocess_info(ServiceType.TIMING_MANAGER, "tm"),
        ]

        workers = subprocess_manager.get_by_type(ServiceType.WORKER)
        rps = subprocess_manager.get_by_type(ServiceType.RECORD_PROCESSOR)
        tms = subprocess_manager.get_by_type(ServiceType.TIMING_MANAGER)

        assert len(workers) == 3
        assert len(rps) == 1
        assert len(tms) == 1
        assert all(w.service_type == ServiceType.WORKER for w in workers)

    def test_get_by_type_returns_empty_for_no_matches(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """get_by_type returns empty list when type not found."""
        subprocess_manager.subprocesses = [
            make_subprocess_info(ServiceType.WORKER, "w1"),
        ]

        result = subprocess_manager.get_by_type(ServiceType.GPU_TELEMETRY_MANAGER)

        assert result == []


# =============================================================================
# Removal and Cleanup Tests
# =============================================================================


class TestSubprocessManagerRemoval:
    """Tests for subprocess removal and cleanup."""

    def test_remove_removes_exact_instance(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """remove removes the specific SubprocessInfo instance."""
        info1 = make_subprocess_info(ServiceType.WORKER, "w1")
        info2 = make_subprocess_info(ServiceType.WORKER, "w2")
        subprocess_manager.subprocesses = [info1, info2]

        subprocess_manager.remove(info1)

        assert info1 not in subprocess_manager.subprocesses
        assert info2 in subprocess_manager.subprocesses
        assert len(subprocess_manager.subprocesses) == 1

    def test_remove_ignores_nonexistent_instance(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """remove does nothing when instance not in list."""
        info1 = make_subprocess_info(ServiceType.WORKER, "w1")
        info_not_tracked = make_subprocess_info(ServiceType.WORKER, "not_tracked")
        subprocess_manager.subprocesses = [info1]

        subprocess_manager.remove(info_not_tracked)

        assert len(subprocess_manager.subprocesses) == 1

    def test_clear_removes_all(self, subprocess_manager: SubprocessManager) -> None:
        """clear empties the subprocess list."""
        subprocess_manager.subprocesses = [
            make_subprocess_info(ServiceType.WORKER, "w1"),
            make_subprocess_info(ServiceType.WORKER, "w2"),
            make_subprocess_info(ServiceType.RECORD_PROCESSOR, "rp1"),
        ]

        subprocess_manager.clear()

        assert subprocess_manager.subprocesses == []


# =============================================================================
# Logger Integration Tests
# =============================================================================


class TestSubprocessManagerLogger:
    """Tests for logger integration."""

    def test_debug_calls_logger_when_available(
        self, subprocess_manager_with_logger: tuple[SubprocessManager, MagicMock]
    ) -> None:
        """_debug delegates to logger.debug when logger exists."""
        manager, mock_logger = subprocess_manager_with_logger

        manager._debug("test message")

        mock_logger.debug.assert_called_once_with("test message")

    def test_debug_is_noop_without_logger(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """_debug does not raise when no logger configured."""
        subprocess_manager._debug("test message")  # Should not raise

    def test_warning_calls_logger_when_available(
        self, subprocess_manager_with_logger: tuple[SubprocessManager, MagicMock]
    ) -> None:
        """_warning delegates to logger.warning when logger exists."""
        manager, mock_logger = subprocess_manager_with_logger

        manager._warning("warning message")

        mock_logger.warning.assert_called_once_with("warning message")

    def test_warning_is_noop_without_logger(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """_warning does not raise when no logger configured."""
        subprocess_manager._warning("warning message")  # Should not raise


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestSubprocessManagerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method",
        [
            param("stop_all", id="stop_all"),
            param("kill_all", id="kill_all"),
        ],
    )  # fmt: skip
    async def test_stop_and_kill_handle_empty_list(
        self, subprocess_manager: SubprocessManager, method: str
    ) -> None:
        """stop_all and kill_all return empty list when no subprocesses."""
        subprocess_manager.subprocesses = []

        result = await getattr(subprocess_manager, method)()

        assert result == []

    @pytest.mark.asyncio
    async def test_spawn_services_zero_replicas_returns_empty(
        self, subprocess_manager: SubprocessManager
    ) -> None:
        """spawn_services with num_replicas=0 creates nothing."""
        with mock_mp_process(MagicMock(spec=Process)):
            result = await subprocess_manager.spawn_services(
                service_type=ServiceType.WORKER,
                num_replicas=0,
            )

            assert result == []
            assert subprocess_manager.subprocesses == []

    @pytest.mark.parametrize(
        "method",
        [
            param("check_alive", id="check_alive"),
            param("get_by_type", id="get_by_type"),
        ],
    )  # fmt: skip
    def test_check_and_get_handle_empty_list(
        self, subprocess_manager: SubprocessManager, method: str
    ) -> None:
        """check_alive and get_by_type handle empty subprocess list."""
        subprocess_manager.subprocesses = []

        if method == "check_alive":
            result = subprocess_manager.check_alive()
        else:
            result = subprocess_manager.get_by_type(ServiceType.WORKER)

        assert result == []

    @pytest.mark.parametrize(
        ("num_workers", "num_rps", "expected_workers", "expected_rps"),
        [
            param(0, 0, 0, 0, id="empty"),
            param(1, 0, 1, 0, id="workers_only"),
            param(0, 1, 0, 1, id="rps_only"),
            param(5, 2, 5, 2, id="mixed"),
            param(10, 3, 10, 3, id="many"),
        ],
    )  # fmt: skip
    def test_filtering_with_mixed_types(
        self,
        subprocess_manager: SubprocessManager,
        num_workers: int,
        num_rps: int,
        expected_workers: int,
        expected_rps: int,
    ) -> None:
        """get_by_type correctly filters various combinations of service types."""
        subprocess_manager.subprocesses = [
            make_subprocess_info(ServiceType.WORKER, f"w{i}")
            for i in range(num_workers)
        ] + [
            make_subprocess_info(ServiceType.RECORD_PROCESSOR, f"rp{i}")
            for i in range(num_rps)
        ]

        workers = subprocess_manager.get_by_type(ServiceType.WORKER)
        rps = subprocess_manager.get_by_type(ServiceType.RECORD_PROCESSOR)

        assert len(workers) == expected_workers
        assert len(rps) == expected_rps
