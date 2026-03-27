# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pytest import param

from aiperf.common.enums import WorkerStartupState
from aiperf.common.messages import (
    DatasetConfiguredNotification,
    DatasetDownloadedNotification,
)
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
from aiperf.config import AIPerfConfig, BenchmarkRun
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
    async def test_dataset_configured_deferred_in_kubernetes_mode(
        self, k8s_worker: Worker
    ) -> None:
        """In K8s mode, dataset config should be stored as pending, not processed immediately."""
        mock_msg = MagicMock()
        mock_msg.client_metadata = MagicMock()
        mock_msg.metadata = MagicMock()

        # Patch _initialize_dataset_client to verify it's NOT called
        k8s_worker._initialize_dataset_client = AsyncMock()

        await k8s_worker._on_dataset_configured(mock_msg)

        # Should have stored as pending
        assert k8s_worker._pending_dataset_config is mock_msg
        # Should NOT have initialized client
        k8s_worker._initialize_dataset_client.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dataset_configured_ignores_stale_benchmark_notification(
        self, k8s_worker: Worker
    ) -> None:
        """K8s workers should ignore dataset notifications from a different benchmark ID."""
        stale_msg = DatasetConfiguredNotification(
            service_id="dataset_manager",
            metadata=DatasetMetadata(
                conversations=[],
                total_turn_count=0,
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
        )
        k8s_worker._initialize_dataset_client = AsyncMock()

        await k8s_worker._on_dataset_configured(stale_msg)

        assert k8s_worker._pending_dataset_config is None
        k8s_worker._initialize_dataset_client.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dataset_configured_immediate_in_local_mode(
        self, local_worker: Worker
    ) -> None:
        """In local mode, dataset config should initialize the client immediately."""
        mock_msg = MagicMock()
        mock_msg.client_metadata = MagicMock()
        mock_msg.metadata = MagicMock()

        # Patch _initialize_dataset_client to verify it IS called
        local_worker._initialize_dataset_client = AsyncMock()

        await local_worker._on_dataset_configured(mock_msg)

        # Should NOT have stored as pending
        assert local_worker._pending_dataset_config is None
        # Should have initialized client immediately
        local_worker._initialize_dataset_client.assert_awaited_once_with(
            mock_msg.client_metadata, mock_msg.metadata
        )

    @pytest.mark.asyncio
    async def test_profile_configure_waits_for_worker_ready(
        self, config: AIPerfConfig
    ) -> None:
        """PROFILE_CONFIGURE should not complete until WorkerReady has been sent."""
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
    async def test_local_worker_marks_ready_on_start(
        self, config: AIPerfConfig
    ) -> None:
        """Local workers should be considered ready immediately after startup."""
        config.runtime.service_run_type = ServiceRunType.MULTIPROCESSING
        worker = Worker(
            run=self._make_run(config),
            service_id="local-worker",
        )
        worker.publish = AsyncMock()
        worker._measure_baseline_rtt = AsyncMock()
        worker.return_dealer_client.send = AsyncMock()

        await worker._send_worker_ready_message()

        assert worker._worker_ready_event.is_set()
        assert [
            call.args[0].startup_state for call in worker.publish.await_args_list
        ] == [
            WorkerStartupState.STARTING,
            WorkerStartupState.ROUTER_PROBING,
            WorkerStartupState.READY,
        ]

    @pytest.mark.asyncio
    async def test_dataset_downloaded_is_idempotent_after_worker_ready(
        self, config: AIPerfConfig
    ) -> None:
        """Repeated dataset-downloaded notifications should not re-run readiness flow."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker.publish = AsyncMock()
        pending_config = MagicMock()
        pending_config.metadata = MagicMock()
        worker._pending_dataset_config = pending_config
        worker._initialize_dataset_client = AsyncMock()
        worker._measure_baseline_rtt = AsyncMock()
        worker.return_dealer_client.send = AsyncMock()

        downloaded = MagicMock()
        downloaded.success = True
        downloaded.client_metadata = MagicMock()
        downloaded.client_metadata.data_file_path = Path(
            f"/aiperf/datasets/aiperf_mmap_{config.artifacts.benchmark_id}/dataset.dat"
        )

        await worker._on_dataset_downloaded(downloaded)
        await worker._on_dataset_downloaded(downloaded)

        worker._initialize_dataset_client.assert_awaited_once()
        worker._measure_baseline_rtt.assert_awaited_once()
        worker.return_dealer_client.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rebroadcasted_config_and_download_do_not_rerun_ready_flow(
        self, config: AIPerfConfig
    ) -> None:
        """Repeated config/download rebroadcasts after ready should be ignored."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker.publish = AsyncMock()
        worker._initialize_dataset_client = AsyncMock()
        worker._measure_baseline_rtt = AsyncMock()
        worker.return_dealer_client.send = AsyncMock()

        dataset_metadata = DatasetMetadata(
            conversations=[],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        configured = DatasetConfiguredNotification(
            service_id="dataset_manager",
            metadata=dataset_metadata,
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
        )
        downloaded = DatasetDownloadedNotification(
            service_id="worker-pod-manager",
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
            success=True,
        )

        await worker._send_worker_ready_message()
        await worker._on_dataset_configured(configured)
        await worker._on_dataset_downloaded(downloaded)
        await worker._on_dataset_configured(configured)
        await worker._on_dataset_downloaded(downloaded)

        worker._initialize_dataset_client.assert_awaited_once_with(
            downloaded.client_metadata,
            dataset_metadata,
        )
        worker._measure_baseline_rtt.assert_awaited_once()
        worker.return_dealer_client.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_k8s_worker_marks_ready_after_downloaded_notification(
        self, config: AIPerfConfig
    ) -> None:
        """Kubernetes workers should set ready only after dataset download completes."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker.publish = AsyncMock()
        pending_config = MagicMock()
        pending_config.metadata = MagicMock()
        worker._pending_dataset_config = pending_config
        worker._worker_ready_event.clear()
        worker._initialize_dataset_client = AsyncMock()
        worker._measure_baseline_rtt = AsyncMock()
        worker.return_dealer_client.send = AsyncMock()

        await worker._send_worker_ready_message()

        downloaded = MagicMock()
        downloaded.success = True
        downloaded.client_metadata = MagicMock()

        await worker._on_dataset_downloaded(downloaded)

        assert worker._worker_ready_event.is_set()
        assert [
            call.args[0].startup_state for call in worker.publish.await_args_list
        ] == [
            WorkerStartupState.STARTING,
            WorkerStartupState.WAITING_FOR_DATASET,
            WorkerStartupState.ROUTER_PROBING,
            WorkerStartupState.READY,
        ]

    @pytest.mark.asyncio
    async def test_k8s_worker_handles_download_before_config_notification(
        self, config: AIPerfConfig
    ) -> None:
        """Kubernetes workers should become ready even if download notification arrives first."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(
            run=self._make_run(config),
            service_id="k8s-worker",
        )
        worker.publish = AsyncMock()
        worker._worker_ready_event.clear()
        worker._initialize_dataset_client = AsyncMock()
        worker._measure_baseline_rtt = AsyncMock()
        worker.return_dealer_client.send = AsyncMock()

        dataset_metadata = DatasetMetadata(
            conversations=[],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        configured = DatasetConfiguredNotification(
            service_id="dataset_manager",
            metadata=dataset_metadata,
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
        )
        downloaded = DatasetDownloadedNotification(
            service_id="worker-pod-manager",
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
            success=True,
        )

        await worker._send_worker_ready_message()
        await worker._on_dataset_downloaded(downloaded)
        await worker._on_dataset_configured(configured)

        worker._initialize_dataset_client.assert_awaited_once_with(
            downloaded.client_metadata,
            dataset_metadata,
        )
        worker._measure_baseline_rtt.assert_awaited_once()
        worker.return_dealer_client.send.assert_awaited_once()
        assert worker._worker_ready_event.is_set()
