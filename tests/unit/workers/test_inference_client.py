# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from pytest import param

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy, RequestContentType
from aiperf.common.models.dataset_models import Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.common.redact import REDACTED_VALUE
from aiperf.plugin.enums import EndpointType, TransportType
from aiperf.workers.inference_client import InferenceClient, detect_transport_from_url


@pytest.fixture
def mock_http_transport_entry():
    """Create a mock transport entry with http/https url_schemes."""
    entry = MagicMock()
    entry.name = TransportType.HTTP.value
    entry.metadata = {"url_schemes": ["http", "https"]}
    return entry


class TestDetectTransportFromUrl:
    """Tests for detect_transport_from_url function."""

    @pytest.fixture(autouse=True)
    def mock_transport_entries(self, mock_http_transport_entry):
        """Mock plugins.list_entries to return http transport with url_schemes."""
        with patch(
            "aiperf.workers.inference_client.plugins.list_entries",
            return_value=[mock_http_transport_entry],
        ):
            yield

    @pytest.mark.parametrize(
        "url,expected_transport",
        [
            param("http://api.example.com:8000", TransportType.HTTP.value, id="http_with_port"),
            param("https://api.example.com:8443", TransportType.HTTP.value, id="https_with_port"),
            param("http://localhost:8000", TransportType.HTTP.value, id="http_localhost"),
            param("http://127.0.0.1:8000", TransportType.HTTP.value, id="http_localhost_ip"),
            param("http://[::1]:8000", TransportType.HTTP.value, id="http_ipv6"),
            param("http://api.example.com", TransportType.HTTP.value, id="http_no_port"),
            param("https://api.example.com", TransportType.HTTP.value, id="https_no_port"),
            param("http://localhost:8000/api/v1/chat", TransportType.HTTP.value, id="with_path"),
            param("http://api.example.com?model=gpt-4&key=value", TransportType.HTTP.value, id="with_query"),
            param("http://user:password@api.example.com:8000", TransportType.HTTP.value, id="with_credentials"),
            param("http://api.example.com#section", TransportType.HTTP.value, id="with_fragment"),
            param("http://api.example.com/path/with%20spaces", TransportType.HTTP.value, id="with_encoded_spaces"),
            param("https://api.openai.com/v1/chat/completions", TransportType.HTTP.value, id="openai_api"),
        ],
    )  # fmt: skip
    def test_http_https_detection(self, url, expected_transport):
        """Test detection of HTTP/HTTPS URLs with various components."""
        result = detect_transport_from_url(url)
        assert result == expected_transport

    @pytest.mark.parametrize(
        "url",
        [
            param("HTTP://api.example.com", id="uppercase_scheme"),
            param("Http://api.example.com", id="mixed_case_scheme"),
            param("hTTp://api.example.com", id="random_case_scheme"),
        ],
    )
    def test_scheme_case_insensitive(self, url):
        """Test that scheme detection is case-insensitive."""
        assert detect_transport_from_url(url) == TransportType.HTTP.value

    @pytest.mark.parametrize(
        "url",
        [
            param("", id="empty_string"),
            param("http://", id="scheme_only"),
            param("api.example.com:8000", id="no_scheme_with_port"),
            param("api.example.com", id="no_scheme_no_port"),
            param("localhost", id="localhost_no_scheme"),
            param("/path/to/file.sock", id="file_path"),
        ],
    )
    def test_edge_cases_default_to_http_or_raise(self, url):
        """Test edge cases return HTTP or raise ValueError."""
        with contextlib.suppress(ValueError):
            assert detect_transport_from_url(url) == TransportType.HTTP.value

    @pytest.mark.parametrize(
        "url",
        [
            param("unknown://api.example.com", id="unknown_scheme"),
            param("ftp://files.example.com", id="ftp_scheme"),
            param("grpc://localhost:50051", id="grpc_scheme"),
        ],
    )
    def test_unregistered_schemes_raise_error(self, url):
        """Test that unregistered schemes raise ValueError."""
        with pytest.raises(ValueError):
            detect_transport_from_url(url)


class TestInferenceClient:
    """Tests for InferenceClient functionality."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo."""
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000/v1/test",
            ),
        )

    @pytest.fixture
    def inference_client(self, model_endpoint, mock_http_transport_entry):
        """Create an InferenceClient instance."""
        mock_transport = MagicMock()
        mock_endpoint = MagicMock()
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}
        mock_endpoint.format_payload.return_value = {}

        def mock_get_class(protocol, name):
            if protocol == "endpoint":
                return lambda **kwargs: mock_endpoint
            if protocol == "transport":
                return lambda **kwargs: mock_transport
            raise ValueError(f"Unknown protocol: {protocol}")

        with (
            patch(
                "aiperf.workers.inference_client.plugins.get_class",
                side_effect=mock_get_class,
            ),
            patch(
                "aiperf.workers.inference_client.plugins.list_entries",
                return_value=[mock_http_transport_entry],
            ),
        ):
            return InferenceClient(
                model_endpoint=model_endpoint, service_id="test-service-id"
            )

    @pytest.mark.asyncio
    async def test_send_request_sets_endpoint_headers(
        self, inference_client, model_endpoint, sample_request_info
    ):
        """Test that send_request sets endpoint_headers on request_info and redacts after transport."""
        model_endpoint.endpoint.api_key = "test-key"
        model_endpoint.endpoint.headers = [("X-Custom", "value")]

        request_info = sample_request_info

        expected_headers = {
            "Authorization": "Bearer test-key",
            "X-Custom": "value",
        }
        inference_client.endpoint.get_endpoint_headers.return_value = expected_headers

        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        await inference_client.send_request(request_info)

        # After send_request, sensitive headers are redacted on request_info
        assert "Authorization" in request_info.endpoint_headers
        assert request_info.endpoint_headers["Authorization"] == REDACTED_VALUE
        assert request_info.endpoint_headers["X-Custom"] == "value"

    @pytest.mark.asyncio
    async def test_send_request_sets_endpoint_params(
        self, inference_client, model_endpoint, sample_request_info
    ):
        """Test that send_request sets endpoint_params on request_info."""
        model_endpoint.endpoint.url_params = {"api-version": "v1", "timeout": "30"}

        request_info = sample_request_info

        expected_params = {"api-version": "v1", "timeout": "30"}
        inference_client.endpoint.get_endpoint_params.return_value = expected_params

        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        await inference_client.send_request(request_info)

        assert request_info.endpoint_params["api-version"] == "v1"
        assert request_info.endpoint_params["timeout"] == "30"

    @pytest.mark.asyncio
    async def test_send_request_calls_transport(
        self,
        inference_client,
        model_endpoint,
        sample_request_info,
        sample_request_record,
    ):
        """Test that send_request delegates to transport."""
        request_info = sample_request_info
        expected_record = sample_request_record

        inference_client.transport.send_request = AsyncMock(
            return_value=expected_record
        )

        record = await inference_client.send_request(request_info)

        inference_client.transport.send_request.assert_called_once()
        call_args = inference_client.transport.send_request.call_args
        assert call_args[0][0] == request_info
        assert record == expected_record

    @pytest.mark.parametrize(
        "raw_payload",
        [
            param(
                {
                    "messages": [{"role": "user", "content": "exact body"}],
                    "temperature": 0.7,
                    "vendor_flag": {"preserve": True},
                },
                id="typical_payload",
            ),
            param({}, id="empty_payload"),
            param(
                {
                    "messages": [{"role": "user", "content": "authored"}],
                    "model": "payload-model",
                    "stream": True,
                    "max_tokens": 17,
                    "temperature": 0.01,
                    "tools": [{"type": "function", "function": {"name": "do_it"}}],
                },
                id="formatter_conflicts",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_send_request_sends_raw_payload_without_endpoint_formatting(
        self, inference_client, sample_request_info, raw_payload
    ):
        """Test that raw_payload turns bypass endpoint payload formatting."""
        sample_request_info.turns = [Turn(role="user", raw_payload=raw_payload)]
        sample_request_info.payload_bytes = None
        inference_client.endpoint.format_payload.return_value = {
            "model": "endpoint-model",
            "stream": False,
            "messages": [{"role": "user", "content": "rewritten"}],
        }
        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        await inference_client.send_request(sample_request_info)

        inference_client.endpoint.format_payload.assert_not_called()
        call_args = inference_client.transport.send_request.call_args
        assert call_args.kwargs["payload"] is sample_request_info.payload_bytes
        assert orjson.loads(call_args.kwargs["payload"]) == raw_payload

    @pytest.mark.asyncio
    async def test_send_request_formats_when_only_earlier_turn_has_raw_payload(
        self, inference_client, sample_request_info
    ):
        """Test raw_payload passthrough is scoped to the current turn."""
        request_info = sample_request_info
        request_info.turns = [
            Turn(
                role="user",
                raw_payload={"messages": [{"role": "user", "content": "old"}]},
            ),
            Turn(role="user", texts=[Text(contents=["current"])]),
        ]
        expected_payload = {"messages": [{"role": "user", "content": "current"}]}
        expected_record = RequestRecord()
        inference_client.endpoint.format_payload.return_value = expected_payload
        inference_client.transport.send_request = AsyncMock(
            return_value=expected_record
        )

        await inference_client.send_request(request_info)

        inference_client.endpoint.format_payload.assert_called_once_with(request_info)
        call_args = inference_client.transport.send_request.call_args
        assert orjson.loads(call_args.kwargs["payload"]) == expected_payload

    @pytest.mark.asyncio
    async def test_send_request_uses_payload_bytes_when_present(
        self, inference_client, sample_request_info
    ):
        """When request_info.payload_bytes is already populated (mmap fast path),
        inference_client passes it straight through — no format_payload, no
        re-encode, and turns may be empty."""
        sample_request_info.payload_bytes = (
            b'{"messages":[{"role":"user","content":"verbatim"}]}'
        )
        sample_request_info.turns = []
        expected_record = RequestRecord()
        inference_client.endpoint.format_payload.return_value = {"rewritten": True}
        inference_client.transport.send_request = AsyncMock(
            return_value=expected_record
        )

        await inference_client.send_request(sample_request_info)

        inference_client.endpoint.format_payload.assert_not_called()
        call_args = inference_client.transport.send_request.call_args
        # Identity check: the same bytes object reaches the transport.
        assert call_args.kwargs["payload"] is sample_request_info.payload_bytes

    @pytest.fixture
    def multipart_inference_client(self, model_endpoint, mock_http_transport_entry):
        model_endpoint.endpoint.request_content_type = (
            RequestContentType.MULTIPART_FORM_DATA
        )
        mock_transport = MagicMock()
        mock_endpoint = MagicMock()
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}
        mock_endpoint.format_payload.return_value = {"rewritten": True}

        def mock_get_class(protocol, name):
            if protocol == "endpoint":
                return lambda **kwargs: mock_endpoint
            if protocol == "transport":
                return lambda **kwargs: mock_transport
            raise ValueError(f"Unknown protocol: {protocol}")

        with (
            patch(
                "aiperf.workers.inference_client.plugins.get_class",
                side_effect=mock_get_class,
            ),
            patch(
                "aiperf.workers.inference_client.plugins.list_entries",
                return_value=[mock_http_transport_entry],
            ),
        ):
            return InferenceClient(
                model_endpoint=model_endpoint, service_id="test-service-id"
            )

    @pytest.mark.asyncio
    async def test_send_request_keeps_multipart_payload_dict_for_transport(
        self, multipart_inference_client, sample_request_info
    ):
        raw_payload = {
            "prompt": "edit this",
            "image": "data:image/png;base64,abc",
        }
        sample_request_info.model_endpoint = multipart_inference_client.model_endpoint
        sample_request_info.turns = [Turn(role="user", raw_payload=raw_payload)]
        sample_request_info.payload_bytes = None
        multipart_inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        await multipart_inference_client.send_request(sample_request_info)

        multipart_inference_client.endpoint.format_payload.assert_not_called()
        call_args = multipart_inference_client.transport.send_request.call_args
        assert call_args.kwargs["payload"] == raw_payload
        assert orjson.loads(sample_request_info.payload_bytes) == raw_payload

    @pytest.mark.asyncio
    async def test_send_request_keeps_multipart_formatter_dict_for_transport(
        self, multipart_inference_client, sample_request_info
    ):
        formatted_payload = {
            "prompt": "edit this",
            "image": "data:image/png;base64,abc",
            "num_inference_steps": 4,
        }
        sample_request_info.model_endpoint = multipart_inference_client.model_endpoint
        sample_request_info.turns = [Turn(role="user", texts=[Text(contents=["edit"])])]
        sample_request_info.payload_bytes = None
        multipart_inference_client.endpoint.format_payload.return_value = (
            formatted_payload
        )
        multipart_inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        await multipart_inference_client.send_request(sample_request_info)

        multipart_inference_client.endpoint.format_payload.assert_called_once_with(
            sample_request_info
        )
        call_args = multipart_inference_client.transport.send_request.call_args
        assert call_args.kwargs["payload"] == formatted_payload
        assert orjson.loads(sample_request_info.payload_bytes) == formatted_payload

    @pytest.mark.asyncio
    async def test_send_request_keeps_multipart_file_tuple_payload_dict_for_transport(
        self, multipart_inference_client, sample_request_info
    ):
        raw_payload = {
            "prompt": "remove the chair",
            "image": ("room.png", b"\x89PNG\r\n", "image/png"),
        }
        sample_request_info.model_endpoint = multipart_inference_client.model_endpoint
        sample_request_info.turns = [Turn(role="user", raw_payload=raw_payload)]
        sample_request_info.payload_bytes = None
        multipart_inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        await multipart_inference_client.send_request(sample_request_info)

        multipart_inference_client.endpoint.format_payload.assert_not_called()
        call_args = multipart_inference_client.transport.send_request.call_args
        assert call_args.kwargs["payload"] == raw_payload
        assert isinstance(call_args.kwargs["payload"], dict)
        assert sample_request_info.payload_bytes is None

    @pytest.mark.asyncio
    async def test_send_request_json_formatter_dict_sets_payload_bytes(
        self, inference_client, sample_request_info
    ):
        formatted_payload = {
            "messages": [{"role": "user", "content": "formatted"}],
            "max_tokens": 8,
        }
        sample_request_info.turns = [Turn(role="user", texts=[Text(contents=["hi"])])]
        sample_request_info.payload_bytes = None
        inference_client.endpoint.format_payload.return_value = formatted_payload
        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        await inference_client.send_request(sample_request_info)

        call_args = inference_client.transport.send_request.call_args
        assert isinstance(call_args.kwargs["payload"], bytes)
        assert call_args.kwargs["payload"] is sample_request_info.payload_bytes
        assert orjson.loads(sample_request_info.payload_bytes) == formatted_payload

    @pytest.mark.asyncio
    async def test_send_request_raises_only_when_turns_and_payload_bytes_both_empty(
        self, inference_client
    ):
        """The empty-turns guard now permits payload_bytes-only RequestInfo."""
        request_info = RequestInfo(
            model_endpoint=inference_client.model_endpoint,
            turns=[],
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="r",
            x_correlation_id="c",
            conversation_id="conv",
        )

        with pytest.raises(ValueError, match="no turns and no payload_bytes"):
            await inference_client.send_request(request_info)

        # With payload_bytes set, no raise.
        request_info.payload_bytes = b'{"messages":[]}'
        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )
        await inference_client.send_request(request_info)

    @pytest.mark.asyncio
    async def test_send_request_populates_metric_inputs(
        self, inference_client, sample_request_info
    ):
        """metric_inputs carries routing fields and inline payload bytes."""
        sample_request_info.payload_bytes = (
            b'{"messages":[{"role":"user","content":"x"}]}'
        )
        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        record = await inference_client.send_request(sample_request_info)

        assert record.metric_inputs is not None
        assert record.metric_inputs.credit_num == sample_request_info.credit_num
        assert record.metric_inputs.credit_phase == sample_request_info.credit_phase
        assert (
            record.metric_inputs.conversation_id == sample_request_info.conversation_id
        )
        assert record.metric_inputs.turn_index == sample_request_info.turn_index
        assert record.metric_inputs.x_request_id == sample_request_info.x_request_id
        assert (
            record.metric_inputs.x_correlation_id
            == sample_request_info.x_correlation_id
        )
        assert (
            record.metric_inputs.credit_issued_ns
            == sample_request_info.credit_issued_ns
        )
        assert record.metric_inputs.agent_depth == sample_request_info.agent_depth
        assert (
            record.metric_inputs.parent_correlation_id
            == sample_request_info.parent_correlation_id
        )
        assert (
            record.metric_inputs.payload_bytes_or_none
            == sample_request_info.payload_bytes
        )

    @pytest.mark.asyncio
    async def test_send_request_metric_inputs_carries_all_routing_fields(
        self, inference_client, sample_request_info
    ):
        """MetricInputs carries every routing field that ``RequestInfo`` carries."""
        sample_request_info.payload_bytes = b'{"messages":[]}'
        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        record = await inference_client.send_request(sample_request_info)

        assert record.metric_inputs is not None
        assert record.metric_inputs.credit_num == sample_request_info.credit_num
        assert (
            record.metric_inputs.conversation_id == sample_request_info.conversation_id
        )
        assert (
            record.metric_inputs.payload_bytes_or_none
            == sample_request_info.payload_bytes
        )

    @pytest.mark.asyncio
    async def test_send_request_drops_payload_bytes_when_from_mmap(
        self, inference_client, sample_request_info
    ):
        """from_mmap=True -> records-process resolves bytes itself; the wire
        MetricInputs.payload_bytes is None to avoid shipping them twice."""
        sample_request_info.payload_bytes = (
            b'{"messages":[{"role":"user","content":"x"}]}'
        )
        sample_request_info.from_mmap = True
        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        record = await inference_client.send_request(sample_request_info)

        assert record.metric_inputs is not None
        # Wire payload_bytes dropped on PAYLOAD_BYTES path.
        assert record.metric_inputs.payload_bytes_or_none is None
        # Routing fields still populated.
        assert (
            record.metric_inputs.conversation_id == sample_request_info.conversation_id
        )
        assert record.metric_inputs.turn_index == sample_request_info.turn_index

    @pytest.mark.asyncio
    async def test_send_request_keeps_payload_bytes_when_not_from_mmap(
        self, inference_client, sample_request_info
    ):
        """from_mmap=False (default) -> wire MetricInputs.payload_bytes carries
        the inline bytes for CONVERSATION-format datasets and error paths."""
        sample_request_info.payload_bytes = (
            b'{"messages":[{"role":"user","content":"x"}]}'
        )
        # from_mmap defaults to False.
        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        record = await inference_client.send_request(sample_request_info)

        assert record.metric_inputs is not None
        assert (
            record.metric_inputs.payload_bytes_or_none
            == sample_request_info.payload_bytes
        )

    def test_enrich_request_record_uses_last_turn_model(self, inference_client):
        """Test _enrich_request_record uses turns[-1] not turns[turn_index].

        In MESSAGE_ARRAY_WITH_RESPONSES mode, turn_list has only 1 element
        but turn_index reflects the actual conversation position (e.g. 3).
        Using turns[turn_index] would raise IndexError.
        """
        turn = Turn(
            texts=[Text(contents=["standalone turn"])],
            role="user",
            model="standalone-model",
        )
        request_info = RequestInfo(
            model_endpoint=inference_client.model_endpoint,
            turns=[turn],
            turn_index=3,
            credit_num=0,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="test-id",
            x_correlation_id="test-corr",
            conversation_id="test-conv",
        )
        record = RequestRecord(
            start_perf_ns=1000,
            timestamp_ns=1000,
            end_perf_ns=2000,
        )

        result = inference_client._finalize_request_record(
            record=record, request_info=request_info
        )

        assert result.model_name == "standalone-model"

    @pytest.mark.parametrize(
        "base_url",
        [
            param("http://127.0.0.1:8000", id="explicit-http"),
            param("https://api.example.com", id="explicit-https"),
        ],
    )  # fmt: skip
    def test_auto_detected_transport_serializes_without_pydantic_warning(
        self, base_url, mock_http_transport_entry
    ):
        """InferenceClient must set transport as a TransportType enum, not a bare str.

        Assigning the raw plugin name string post-validation triggers
        PydanticSerializationUnexpectedValue at model_dump() time because the
        field is typed TransportType | None but holds a plain str.
        """
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_urls=[base_url],
            ),
        )

        def mock_get_class(protocol, name):
            return MagicMock()

        with (
            patch(
                "aiperf.workers.inference_client.plugins.get_class",
                side_effect=mock_get_class,
            ),
            patch(
                "aiperf.workers.inference_client.plugins.list_entries",
                return_value=[mock_http_transport_entry],
            ),
        ):
            InferenceClient(model_endpoint=model_endpoint, service_id="test-svc")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model_endpoint.model_dump()

        pydantic_warnings = [
            w
            for w in captured
            if "PydanticSerializationUnexpectedValue" in str(w.message)
        ]
        assert not pydantic_warnings, (
            f"Unexpected Pydantic serialization warnings for {base_url!r}: {pydantic_warnings}"
        )
