# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pytest import param

from aiperf.common.enums import MessageType, WorkerStartupState
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.models import (
    Conversation,
    DatasetMetadata,
    MemoryMapClientMetadata,
    ParsedResponse,
    ProcessHealth,
    ReasoningResponseData,
    RequestRecord,
    SSEMessage,
    TextResponseData,
)
from aiperf.common.pod_lifecycle_structs import (
    GroupDatasetReady,
    GroupDatasetStateSnapshot,
    GroupPeerShutdown,
    GroupWorkerHealth,
    GroupWorkerStartupState,
)
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.credit.messages import (
    WorkerConnected,
    WorkerDispatchable,
    WorkerUndispatchable,
)
from aiperf.credit.structs import Credit, CreditContext
from aiperf.plugin.enums import DatasetSamplingStrategy, ServiceRunType
from aiperf.workers.worker import Worker
from tests.harness.fake_communication import FakeCommunication as FakeCommunication
from tests.harness.fake_service_manager import FakeServiceManager as FakeServiceManager
from tests.harness.fake_tokenizer import FakeTokenizer
from tests.harness.fake_transport import FakeTransport as FakeTransport

_STUB_PROCESS_HEALTH = ProcessHealth(
    create_time=0.0, uptime=1.0, cpu_usage=0.0, memory_usage=0
)


@pytest.fixture
def mock_worker(
    run: BenchmarkRun,
    fake_tokenizer: FakeTokenizer,
    skip_service_registration,
    mock_psutil_process,
):
    """Create a constructed MockWorker without starting the lifecycle.

    These tests exercise internal worker methods directly and do not need the
    full lifecycle. Avoid starting the worker here so fixture teardown does not
    depend on the fake in-process comms harness.
    """
    worker = Worker(
        run=run,
        service_id="mock-service-id",
    )
    worker._measure_baseline_rtt = AsyncMock()
    worker.get_process_health = Mock(return_value=_STUB_PROCESS_HEALTH)
    worker.get_pss_memory = Mock(return_value=None)
    return worker


@pytest.mark.asyncio
class TestWorker:
    async def test_process_response(
        self, monkeypatch, mock_worker, sample_request_record
    ):
        """Ensure process_response extracts text correctly from RequestRecord."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text="Hello, world!"),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(sample_request_record)
        assert turn.texts[0].contents == ["Hello, world!"]

    async def test_process_response_empty(
        self, monkeypatch, mock_worker, sample_request_record
    ):
        """Ensure process_response handles empty responses correctly."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text=""),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(sample_request_record)
        assert turn is None

    async def test_process_response_reasoning_extracts_content(
        self, monkeypatch, mock_worker
    ):
        """Ensure process_response extracts content from reasoning responses."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=ReasoningResponseData(
                reasoning="Let me think...",
                content="The answer is 42.",
            ),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(RequestRecord())
        assert turn.texts[0].contents == ["The answer is 42."]

    async def test_process_response_reasoning_only_returns_none(
        self, monkeypatch, mock_worker
    ):
        """Ensure process_response returns None for reasoning-only responses (no content)."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=ReasoningResponseData(
                reasoning="Let me think about this...",
                content=None,
            ),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(RequestRecord())
        assert turn is None

    async def test_process_response_mixed_reasoning_and_text_combines_content(
        self, monkeypatch, mock_worker
    ):
        """Ensure process_response combines text and reasoning content."""
        mock_parsed_responses = [
            ParsedResponse(
                perf_ns=0,
                data=TextResponseData(text="Hello"),
            ),
            ParsedResponse(
                perf_ns=1,
                data=ReasoningResponseData(
                    reasoning="Thinking...",
                    content="World",
                ),
            ),
        ]
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=mock_parsed_responses)
        monkeypatch.setattr(mock_worker.inference_client, "endpoint", mock_endpoint)
        turn = await mock_worker._process_response(RequestRecord())
        assert turn.texts[0].contents == ["HelloWorld"]


# --- FirstToken Callback Test Helpers ---


def create_first_token_callback(worker: Worker):
    """Create a first token callback that mirrors Worker implementation.

    This callback uses endpoint.parse_response to check if an SSE message
    contains meaningful content.

    Returns:
        Async callback function (ttft_ns, message) -> bool
    """

    async def first_token_callback(ttft_ns: int, message: SSEMessage) -> bool:
        parsed = worker.inference_client.endpoint.parse_response(message)
        return parsed is not None and parsed.data is not None

    return first_token_callback


def setup_mock_endpoint(worker: Worker, monkeypatch, parse_response_return):
    """Setup mock endpoint with specified parse_response return value.

    Args:
        worker: MockWorker instance
        monkeypatch: pytest monkeypatch fixture
        parse_response_return: Return value or side_effect for parse_response
    """
    mock_endpoint = Mock()
    if isinstance(parse_response_return, list):
        mock_endpoint.parse_response = Mock(side_effect=parse_response_return)
    else:
        mock_endpoint.parse_response = Mock(return_value=parse_response_return)
    mock_endpoint.extract_response_data = Mock()  # Should NOT be called
    monkeypatch.setattr(worker.inference_client, "endpoint", mock_endpoint)
    return mock_endpoint


@pytest.mark.asyncio
class TestWorkerFirstTokenCallback:
    """Test suite for Worker's first_token_callback logic."""

    @pytest.mark.parametrize(
        "parse_return,expected_result,description",
        [
            # Meaningful content - should return True
            pytest.param(
                ParsedResponse(
                    perf_ns=100_000_000, data=TextResponseData(text="Hello")
                ),
                True,
                "meaningful text content",
                id="meaningful_content",
            ),
            # None response - should return False
            pytest.param(
                None,
                False,
                "parse_response returns None",
                id="none_response",
            ),
            # ParsedResponse with data=None (usage only) - should return False
            pytest.param(
                ParsedResponse(
                    perf_ns=100_000_000,
                    data=None,
                    usage={"prompt_tokens": 10, "completion_tokens": 0},
                ),
                False,
                "usage-only response with data=None",
                id="none_data",
            ),
        ],
    )
    async def test_callback_return_value(
        self, monkeypatch, mock_worker, parse_return, expected_result, description
    ):
        """Test callback returns correct bool based on parse_response result."""
        setup_mock_endpoint(mock_worker, monkeypatch, parse_return)
        callback = create_first_token_callback(mock_worker)

        test_message = SSEMessage(perf_ns=100_000_000)
        result = await callback(50_000_000, test_message)

        assert result is expected_result, f"Failed for: {description}"

    async def test_callback_finds_first_meaningful_content_after_junk(
        self, monkeypatch, mock_worker
    ):
        """Test callback correctly identifies first meaningful content after junk messages."""
        parse_returns = [
            None,  # First: junk
            ParsedResponse(perf_ns=200_000_000, data=None),  # Second: usage only
            ParsedResponse(  # Third: actual content
                perf_ns=300_000_000,
                data=TextResponseData(text="Finally some content!"),
            ),
        ]

        setup_mock_endpoint(mock_worker, monkeypatch, parse_returns)
        callback = create_first_token_callback(mock_worker)

        messages = [SSEMessage(perf_ns=i * 100_000_000) for i in range(1, 4)]
        results = [await callback(msg.perf_ns, msg) for msg in messages]

        assert results == [False, False, True]


# --- Fixture for CreditContext ---


@pytest.fixture
def sample_credit_context() -> CreditContext:
    """Create a sample CreditContext for testing."""
    return CreditContext(
        credit=Credit(
            id=1,
            phase="profiling",
            conversation_id="test-conv-123",
            x_correlation_id="test-correlation-id",
            turn_index=0,
            num_turns=1,
            issued_at_ns=1000000,
        ),
        drop_perf_ns=2000000,
    )


# --- RetrieveConversation Tests ---


@pytest.mark.asyncio
class TestRetrieveConversation:
    """Test suite for Worker's _retrieve_conversation method."""

    async def test_returns_from_dataset_client_when_available(
        self, mock_worker, sample_credit_context
    ):
        """When _dataset_client is set, should return conversation from it."""
        expected_conversation = Conversation(session_id="test-conv-123", turns=[])
        mock_client = AsyncMock()
        mock_client.get_conversation = AsyncMock(return_value=expected_conversation)
        mock_worker._dataset_client = mock_client

        result = await mock_worker._retrieve_conversation(
            conversation_id="test-conv-123",
            credit_context=sample_credit_context,
        )

        assert result == expected_conversation
        mock_client.get_conversation.assert_called_once_with("test-conv-123")

    async def test_raises_cancelled_error_when_stop_requested_and_no_client(
        self, mock_worker, sample_credit_context
    ):
        """When _dataset_client is None and stop_requested, should raise CancelledError."""
        mock_worker._dataset_client = None
        mock_worker.stop_requested = True

        with pytest.raises(asyncio.CancelledError, match="Stop requested"):
            await mock_worker._retrieve_conversation(
                conversation_id="test-conv-123",
                credit_context=sample_credit_context,
            )

    async def test_falls_back_to_dataset_manager_when_no_client_and_not_stopping(
        self, monkeypatch, mock_worker, sample_credit_context
    ):
        """When _dataset_client is None and not stopping, should request from DatasetManager."""
        mock_worker._dataset_client = None
        expected_conversation = Conversation(session_id="test-conv-123", turns=[])
        mock_fallback = AsyncMock(return_value=expected_conversation)
        monkeypatch.setattr(
            mock_worker, "_request_conversation_from_dataset_manager", mock_fallback
        )

        result = await mock_worker._retrieve_conversation(
            conversation_id="test-conv-123",
            credit_context=sample_credit_context,
        )

        assert result == expected_conversation
        mock_fallback.assert_called_once_with("test-conv-123", sample_credit_context)


class TestKubernetesMode:
    """Test Kubernetes-specific behavior in Worker."""

    @staticmethod
    def _make_run(config: AIPerfConfig) -> BenchmarkRun:
        from pathlib import Path

        return BenchmarkRun(
            benchmark_id="test",
            cfg=config,
            artifact_dir=Path("/tmp/test"),
        )

    @pytest.fixture
    def k8s_worker(
        self,
        config: AIPerfConfig,
        fake_tokenizer: FakeTokenizer,
        skip_service_registration,
        mock_psutil_process,
    ) -> Worker:
        """Create a Worker in Kubernetes mode."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker._pod_index = "0"
        worker._measure_baseline_rtt = AsyncMock()
        worker.get_process_health = Mock(return_value=_STUB_PROCESS_HEALTH)
        worker.get_pss_memory = Mock(return_value=None)
        return worker

    @pytest.fixture
    def local_worker(
        self,
        config: AIPerfConfig,
        fake_tokenizer: FakeTokenizer,
        skip_service_registration,
        mock_psutil_process,
    ) -> Worker:
        """Create a Worker in local (multiprocessing) mode."""
        config.runtime.service_run_type = ServiceRunType.MULTIPROCESSING
        worker = Worker(
            run=self._make_run(config),
            service_id="local-worker",
        )
        worker._measure_baseline_rtt = AsyncMock()
        worker.get_process_health = Mock(return_value=_STUB_PROCESS_HEALTH)
        worker.get_pss_memory = Mock(return_value=None)
        return worker

    @pytest.mark.parametrize(
        "run_type,expected",
        [
            param(ServiceRunType.KUBERNETES, True, id="kubernetes"),
            param(ServiceRunType.MULTIPROCESSING, False, id="multiprocessing"),
        ],
    )  # fmt: skip
    def test_is_kubernetes_mode(
        self,
        config: AIPerfConfig,
        run_type: str,
        expected: bool,
    ) -> None:
        """_is_kubernetes_mode should return True only for KUBERNETES run type."""
        config.runtime.service_run_type = run_type
        worker = Worker(
            run=self._make_run(config),
            service_id="test-worker",
        )
        assert worker._is_kubernetes_mode() is expected

    @pytest.mark.asyncio
    async def test_k8s_worker_does_not_subscribe_to_dataset_configured_notification(
        self, k8s_worker: Worker
    ) -> None:
        """Group-managed workers should not subscribe to global dataset broadcasts."""
        k8s_worker.sub_client.subscribe_all = AsyncMock()
        k8s_worker.sub_client.subscribe = AsyncMock()

        await k8s_worker._setup_on_message_hooks()

        subscriptions = k8s_worker.sub_client.subscribe_all.await_args.args[0]
        assert MessageType.DATASET_CONFIGURED_NOTIFICATION not in subscriptions

    @pytest.mark.asyncio
    async def test_dataset_configured_is_ignored_in_kubernetes_mode(
        self, k8s_worker: Worker
    ) -> None:
        """Direct dataset broadcasts should not drive group-managed worker startup."""
        mock_msg = MagicMock()
        mock_msg.client_metadata = MagicMock()
        mock_msg.metadata = MagicMock()
        k8s_worker._initialize_dataset_client = AsyncMock()
        k8s_worker._complete_group_startup_flow = AsyncMock()

        await k8s_worker._on_dataset_configured(mock_msg)

        k8s_worker._initialize_dataset_client.assert_not_awaited()
        k8s_worker._complete_group_startup_flow.assert_not_awaited()
        assert not k8s_worker._dataset_configured_event.is_set()

    @pytest.mark.asyncio
    async def test_dataset_configured_ignores_stale_benchmark_notification(
        self, k8s_worker: Worker
    ) -> None:
        """K8s workers should ignore stale dataset broadcasts without affecting startup."""
        stale_msg = DatasetConfiguredNotification(
            service_id="dataset_manager",
            metadata=DatasetMetadata(
                conversations=[],
                sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            ),
            client_metadata=MemoryMapClientMetadata(
                data_file_path=Path(
                    "/aiperf/datasets/aiperf_mmap_other-benchmark-id/dataset.dat.zst"
                ),
                index_file_path=Path(
                    "/aiperf/datasets/aiperf_mmap_other-benchmark-id/index.dat.zst"
                ),
                compressed=True,
            ),
            benchmark_generation="other-benchmark-id",
            dataset_generation="other-benchmark-id:dataset",
        )
        k8s_worker._initialize_dataset_client = AsyncMock()

        await k8s_worker._on_dataset_configured(stale_msg)

        k8s_worker._initialize_dataset_client.assert_not_awaited()
        assert not k8s_worker._dataset_configured_event.is_set()

    @pytest.mark.asyncio
    async def test_dataset_configured_ignored_in_group_managed_local_mode(
        self, local_worker: Worker
    ) -> None:
        """Group-managed local workers ignore the global dataset broadcast."""
        mock_msg = MagicMock()
        mock_msg.client_metadata = MagicMock()
        mock_msg.metadata = MagicMock()

        local_worker._initialize_dataset_client = AsyncMock()

        await local_worker._on_dataset_configured(mock_msg)

        local_worker._initialize_dataset_client.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_profile_configure_waits_for_worker_ready(
        self, config: AIPerfConfig
    ) -> None:
        """PROFILE_CONFIGURE should not complete until WorkerDispatchable has been sent."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker.event_loop_monitor.start = Mock()
        worker._memory_profiler.start = Mock()
        worker._dataset_configured_event.set()
        worker._worker_ready_event.clear()

        configure_task = asyncio.create_task(
            worker._on_profile_configure_command(MagicMock())
        )
        await asyncio.sleep(0)

        assert not configure_task.done()

        worker._worker_ready_event.set()
        await configure_task

    @pytest.mark.asyncio
    async def test_local_worker_defers_readiness_to_group_startup_flow(
        self, config: AIPerfConfig
    ) -> None:
        """Group-managed local workers defer readiness until group dataset is ready."""
        config.runtime.service_run_type = ServiceRunType.MULTIPROCESSING
        worker = Worker(
            run=self._make_run(config),
            service_id="local-worker",
        )
        worker.publish = AsyncMock()
        worker._measure_baseline_rtt = AsyncMock()
        worker.return_dealer_client.send = AsyncMock()

        await worker._send_worker_ready_message()

        assert not worker._worker_ready_event.is_set()
        assert worker._startup_state == WorkerStartupState.WAITING_FOR_DATASET

    @pytest.mark.asyncio
    async def test_local_worker_skips_global_message_bus_probe_in_group_managed_mode(
        self, local_worker: Worker
    ) -> None:
        """Group-managed local workers skip the global PUB/SUB probe."""
        local_worker._run_connection_probes = AsyncMock()

        await local_worker._wait_for_successful_probe()

        local_worker._run_connection_probes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_k8s_worker_skips_global_message_bus_probe_during_startup(
        self, k8s_worker: Worker
    ) -> None:
        """Group-managed workers should not block startup on the global PUB/SUB probe."""
        k8s_worker._run_connection_probes = AsyncMock()

        await k8s_worker._wait_for_successful_probe()

        k8s_worker._run_connection_probes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_k8s_worker_marks_connected_before_dataset_ready(
        self, k8s_worker: Worker
    ) -> None:
        """K8s workers should connect to the router before dataset readiness."""
        k8s_worker.return_dealer_client.send = AsyncMock()
        k8s_worker._query_pod_dataset_state = AsyncMock(return_value=None)

        await k8s_worker._send_worker_ready_message()

        sent_messages = [
            call.args[0]
            for call in k8s_worker.return_dealer_client.send.await_args_list
        ]
        assert isinstance(sent_messages[0], WorkerConnected)
        assert not any(
            isinstance(message, WorkerDispatchable) for message in sent_messages
        )
        assert k8s_worker._dataset_state_retry_task is not None
        assert not k8s_worker._worker_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_k8s_worker_initializes_from_group_local_dataset_ready(
        self, config: AIPerfConfig
    ) -> None:
        """Group-managed workers should initialize directly from pod-local dataset readiness."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker._pod_index = "0"
        worker.return_dealer_client.send = AsyncMock()
        worker._query_pod_dataset_state = AsyncMock(return_value=None)
        worker._initialize_dataset_client = AsyncMock()

        await worker._on_dataset_ready(
            GroupDatasetReady(
                service_id="worker-pod-manager",
                data_file_path=f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/dataset.dat",
                index_file_path=f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/index.dat",
                conversation_count=4,
                total_size_bytes=1024,
                pod_index="0",
                success=True,
            )
        )

        worker._initialize_dataset_client.assert_awaited_once()
        worker._query_pod_dataset_state.assert_not_awaited()
        assert isinstance(
            worker.return_dealer_client.send.await_args_list[-1].args[0],
            WorkerDispatchable,
        )
        assert worker._worker_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_local_worker_suppresses_dataset_broadcast_in_group_managed_mode(
        self, local_worker: Worker
    ) -> None:
        """Group-managed local workers suppress the global dataset broadcast subscription."""
        local_worker.sub_client.subscribe_all = AsyncMock()
        local_worker.sub_client.subscribe = AsyncMock()

        await local_worker._setup_on_message_hooks()

        subscriptions = local_worker.sub_client.subscribe_all.await_args.args[0]
        assert MessageType.DATASET_CONFIGURED_NOTIFICATION not in subscriptions

    @pytest.mark.asyncio
    async def test_k8s_worker_becomes_dispatchable_after_query_ready(
        self, k8s_worker: Worker
    ) -> None:
        """K8s workers should become dispatchable from pod-local current state."""
        k8s_worker.return_dealer_client.send = AsyncMock()
        k8s_worker._query_pod_dataset_state = AsyncMock(
            return_value=GroupDatasetStateSnapshot(
                rid="rid-1",
                service_id="pod-manager",
                benchmark_generation="gen-1",
                dataset_generation="data-1",
                data_file_path="/aiperf/datasets/dataset.dat",
                index_file_path="/aiperf/datasets/index.dat",
                conversation_count=4,
                total_size_bytes=1024,
                ready=True,
            )
        )
        k8s_worker._initialize_dataset_client = AsyncMock()

        await k8s_worker._complete_group_startup_flow()

        k8s_worker._initialize_dataset_client.assert_awaited_once()
        assert isinstance(
            k8s_worker.return_dealer_client.send.await_args_list[-1].args[0],
            WorkerDispatchable,
        )
        assert k8s_worker._worker_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_k8s_worker_retries_dataset_state_until_ready(
        self, k8s_worker: Worker
    ) -> None:
        """K8s workers should keep polling pod-local dataset state until they become ready."""
        snapshots = [
            None,
            GroupDatasetStateSnapshot(
                rid="rid-1",
                service_id="pod-manager",
                benchmark_generation="gen-1",
                dataset_generation="data-1",
                data_file_path="/aiperf/datasets/dataset.dat",
                index_file_path="/aiperf/datasets/index.dat",
                conversation_count=4,
                total_size_bytes=1024,
                ready=True,
            ),
        ]
        k8s_worker.return_dealer_client.send = AsyncMock()
        k8s_worker._query_pod_dataset_state = AsyncMock(side_effect=snapshots)
        k8s_worker._initialize_dataset_client = AsyncMock()

        await asyncio.wait_for(
            k8s_worker._retry_group_dataset_state_until_ready(), timeout=2.5
        )

        assert k8s_worker._query_pod_dataset_state.await_count >= 2
        k8s_worker._initialize_dataset_client.assert_awaited_once()
        assert k8s_worker._worker_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_k8s_health_checks_use_pod_lifecycle_channel(
        self, k8s_worker: Worker
    ) -> None:
        """Kubernetes workers should send health snapshots directly to WorkerGroupManager."""
        k8s_worker.publish = AsyncMock()
        k8s_worker.pod_lifecycle_dealer_client.send = AsyncMock()

        await k8s_worker._health_check_task()

        k8s_worker.publish.assert_not_awaited()
        sent = k8s_worker.pod_lifecycle_dealer_client.send.await_args.args[0]
        assert isinstance(sent, GroupWorkerHealth)
        assert sent.service_id == "k8s-worker"
        assert sent.task_total == k8s_worker.task_stats.total

    @pytest.mark.asyncio
    async def test_k8s_shutdown_notifies_pod_manager(self, k8s_worker: Worker) -> None:
        """Kubernetes workers should revoke dispatchability before shutdown."""
        k8s_worker.pod_lifecycle_dealer_client.send = AsyncMock()
        k8s_worker.return_dealer_client.send = AsyncMock()

        await k8s_worker._send_worker_shutdown_message()

        lifecycle_sent = k8s_worker.pod_lifecycle_dealer_client.send.await_args_list
        assert isinstance(lifecycle_sent[-1].args[0], GroupPeerShutdown)
        assert lifecycle_sent[-1].args[0].service_id == "k8s-worker"
        sent_messages = [
            call.args[0]
            for call in k8s_worker.return_dealer_client.send.await_args_list
        ]
        assert isinstance(sent_messages[0], WorkerUndispatchable)

    @pytest.mark.asyncio
    async def test_k8s_worker_emits_ready_transition_once_when_snapshot_and_ready_race(
        self, config: AIPerfConfig
    ) -> None:
        """Concurrent pod-local ready signals should emit a single dispatchable/READY transition."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker._pod_index = "0"
        worker.return_dealer_client.send = AsyncMock()
        worker.pod_lifecycle_dealer_client.send = AsyncMock()

        init_started = asyncio.Event()
        release_init = asyncio.Event()

        async def slow_initialize(*args, **kwargs) -> None:
            init_started.set()
            await release_init.wait()
            worker._dataset_configured_event.set()

        worker._initialize_dataset_client = AsyncMock(side_effect=slow_initialize)

        snapshot = GroupDatasetStateSnapshot(
            rid="rid-1",
            service_id="worker-pod-manager",
            benchmark_generation="gen-1",
            dataset_generation="data-1",
            data_file_path=f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/dataset.dat",
            index_file_path=f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/index.dat",
            ready=True,
        )
        dataset_ready = GroupDatasetReady(
            service_id="worker-pod-manager",
            data_file_path=f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/dataset.dat",
            index_file_path=f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/index.dat",
            conversation_count=4,
            total_size_bytes=1024,
            pod_index="0",
            success=True,
        )

        snapshot_task = asyncio.create_task(
            worker._complete_group_startup_flow(snapshot)
        )
        await init_started.wait()
        dataset_ready_task = asyncio.create_task(
            worker._on_dataset_ready(dataset_ready)
        )
        await asyncio.sleep(0)
        release_init.set()
        await asyncio.gather(snapshot_task, dataset_ready_task)

        worker._initialize_dataset_client.assert_awaited_once()
        dispatchable_messages = [
            call.args[0]
            for call in worker.return_dealer_client.send.await_args_list
            if isinstance(call.args[0], WorkerDispatchable)
        ]
        ready_state_messages = [
            call.args[0]
            for call in worker.pod_lifecycle_dealer_client.send.await_args_list
            if isinstance(call.args[0], GroupWorkerStartupState)
            and call.args[0].startup_state == str(WorkerStartupState.READY)
        ]
        assert len(dispatchable_messages) == 1
        assert len(ready_state_messages) == 1
        assert worker._worker_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_dataset_ready_is_idempotent_after_worker_ready(
        self, config: AIPerfConfig
    ) -> None:
        """Repeated pod-local dataset snapshots should not rerun readiness flow."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker._pod_index = "0"
        worker._initialize_dataset_client = AsyncMock()
        worker.return_dealer_client.send = AsyncMock()

        snapshot = GroupDatasetStateSnapshot(
            rid="rid-1",
            service_id="worker-pod-manager",
            benchmark_generation="gen-1",
            dataset_generation="data-1",
            data_file_path=f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/dataset.dat",
            index_file_path=f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/index.dat",
            ready=True,
        )

        await worker._complete_group_startup_flow(snapshot)
        await worker._complete_group_startup_flow(snapshot)

        worker._initialize_dataset_client.assert_awaited_once()
        assert worker._worker_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_rebroadcasted_config_after_ready_is_ignored(
        self, config: AIPerfConfig
    ) -> None:
        """Repeated config notifications after ready should be ignored."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker._worker_ready_event.set()
        worker._initialize_dataset_client = AsyncMock()

        configured = DatasetConfiguredNotification(
            service_id="dataset_manager",
            metadata=DatasetMetadata(
                conversations=[],
                sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            ),
            client_metadata=MemoryMapClientMetadata(
                data_file_path=Path(
                    f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/dataset.dat"
                ),
                index_file_path=Path(
                    f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/index.dat"
                ),
                conversation_count=0,
                total_size_bytes=0,
            ),
            benchmark_generation="gen-1",
            dataset_generation="data-1",
        )

        await worker._on_dataset_configured(configured)

        worker._initialize_dataset_client.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_k8s_worker_ignores_downloaded_notification_from_other_pod(
        self, config: AIPerfConfig
    ) -> None:
        """Kubernetes workers should ignore download notifications from a different pod."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker._pod_index = "0"
        worker._initialize_dataset_client = AsyncMock()
        worker._query_pod_dataset_state = AsyncMock(return_value=None)

        wrong_pod_download = GroupDatasetReady(
            service_id="worker-pod-manager",
            data_file_path=f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/dataset.dat",
            index_file_path=f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/index.dat",
            conversation_count=0,
            total_size_bytes=0,
            pod_index="16",
            success=True,
        )

        await worker._on_dataset_ready(wrong_pod_download)

        worker._initialize_dataset_client.assert_not_awaited()
        worker._query_pod_dataset_state.assert_not_awaited()
        assert not worker._worker_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_k8s_worker_waits_for_ready_snapshot_before_dispatchable(
        self, config: AIPerfConfig
    ) -> None:
        """Kubernetes workers should not become dispatchable from a non-ready snapshot."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker.return_dealer_client.send = AsyncMock()
        worker._initialize_dataset_client = AsyncMock()

        await worker._complete_group_startup_flow(
            GroupDatasetStateSnapshot(
                rid="rid-1",
                service_id="worker-pod-manager",
                benchmark_generation="gen-1",
                dataset_generation="data-1",
                ready=False,
            )
        )

        worker._initialize_dataset_client.assert_not_awaited()
        assert not any(
            isinstance(call.args[0], WorkerDispatchable)
            for call in worker.return_dealer_client.send.await_args_list
        )
        assert not worker._worker_ready_event.is_set()
