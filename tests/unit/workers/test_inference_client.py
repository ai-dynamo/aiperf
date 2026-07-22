# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from pytest import param

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
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
from aiperf.workers.session_routing import (
    DynamoHeadersRouting,
    DynamoNvextRouting,
    SessionIdHeaderRouting,
    SmgRoutingKeyRouting,
)

_ROUTING_CLASSES = {
    "dynamo_headers": DynamoHeadersRouting,
    "dynamo_nvext": DynamoNvextRouting,
    "smg_routing_key": SmgRoutingKeyRouting,
    "session_id_header": SessionIdHeaderRouting,
}


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
            return_value=RequestRecord(request_info=sample_request_info)
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
            return_value=RequestRecord(request_info=sample_request_info)
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

    @pytest.mark.asyncio
    async def test_send_request_sends_raw_payload_without_endpoint_formatting(
        self, inference_client, sample_request_info
    ):
        """Test that raw_payload turns bypass endpoint payload formatting."""
        raw_payload = {
            "messages": [{"role": "user", "content": "exact body"}],
            "temperature": 0.7,
            "vendor_flag": {"preserve": True},
        }
        request_info = sample_request_info
        request_info.turns = [Turn(role="user", raw_payload=raw_payload)]
        expected_record = RequestRecord(request_info=request_info)
        inference_client.endpoint.format_payload.return_value = {"rewritten": True}
        inference_client.transport.send_request = AsyncMock(
            return_value=expected_record
        )

        await inference_client.send_request(request_info)

        inference_client.endpoint.format_payload.assert_not_called()
        call_args = inference_client.transport.send_request.call_args
        # raw_payload bypasses endpoint formatting but is canonicalised to
        # bytes before transport dispatch (so the transport skips its own
        # orjson.dumps and the record carries the exact wire payload).
        assert call_args.kwargs["payload"] == orjson.dumps(raw_payload)

    @pytest.mark.asyncio
    async def test_send_request_sends_empty_raw_payload_without_formatting(
        self, inference_client, sample_request_info
    ):
        """Test that an empty raw_payload is still a verbatim payload."""
        raw_payload = {}
        request_info = sample_request_info
        request_info.turns = [Turn(role="user", raw_payload=raw_payload)]
        expected_record = RequestRecord(request_info=request_info)
        inference_client.endpoint.format_payload.return_value = {"rewritten": True}
        inference_client.transport.send_request = AsyncMock(
            return_value=expected_record
        )

        await inference_client.send_request(request_info)

        inference_client.endpoint.format_payload.assert_not_called()
        call_args = inference_client.transport.send_request.call_args
        assert call_args.kwargs["payload"] == orjson.dumps(raw_payload)

    @pytest.mark.asyncio
    async def test_send_request_preserves_raw_payload_formatter_conflicts(
        self, inference_client, sample_request_info
    ):
        """Test raw_payload top-level fields are not overwritten by endpoint defaults."""
        raw_payload = {
            "messages": [{"role": "user", "content": "authored"}],
            "model": "payload-model",
            "stream": True,
            "max_tokens": 17,
            "temperature": 0.01,
            "tools": [{"type": "function", "function": {"name": "do_it"}}],
        }
        request_info = sample_request_info
        request_info.turns = [Turn(role="user", raw_payload=raw_payload)]
        expected_record = RequestRecord(request_info=request_info)
        inference_client.endpoint.format_payload.return_value = {
            "model": "endpoint-model",
            "stream": False,
            "messages": [{"role": "user", "content": "rewritten"}],
        }
        inference_client.transport.send_request = AsyncMock(
            return_value=expected_record
        )

        await inference_client.send_request(request_info)

        call_args = inference_client.transport.send_request.call_args
        # Verbatim raw_payload, canonicalised to bytes (no endpoint-default
        # overwrites of the authored top-level fields).
        assert call_args.kwargs["payload"] == orjson.dumps(raw_payload)
        sent = orjson.loads(call_args.kwargs["payload"])
        assert sent["model"] == "payload-model"
        assert sent["stream"] is True
        assert sent["messages"][0]["content"] == "authored"

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
        expected_record = RequestRecord(request_info=request_info)
        inference_client.endpoint.format_payload.return_value = expected_payload
        inference_client.transport.send_request = AsyncMock(
            return_value=expected_record
        )

        await inference_client.send_request(request_info)

        inference_client.endpoint.format_payload.assert_called_once_with(request_info)
        call_args = inference_client.transport.send_request.call_args
        assert call_args.kwargs["payload"] == orjson.dumps(expected_payload)

    @pytest.mark.asyncio
    async def test_send_request_raises_on_empty_turns(self, inference_client):
        """Test that send_request raises ValueError when turns is empty."""
        request_info = RequestInfo(
            model_endpoint=inference_client.model_endpoint,
            turns=[],
            turn_index=0,
            credit_num=42,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="test-id",
            x_correlation_id="test-corr",
            conversation_id="test-conv",
        )

        with pytest.raises(ValueError, match="no turns"):
            await inference_client.send_request(request_info)

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
            request_info=request_info,
            start_perf_ns=1000,
            timestamp_ns=1000,
            end_perf_ns=2000,
        )

        result = inference_client._finalize_request_record(
            record=record, request_info=request_info
        )

        assert result.model_name == "standalone-model"

    def test_finalize_request_record_hoists_per_turn_scalars(self, inference_client):
        """max_tokens / audio_duration_seconds must be hoisted from the turns
        onto the RecordContext so record metrics (requested_osl,
        audio_duration) can read them off the record after the downcast.
        """
        turn = Turn(
            texts=[Text(contents=["hoisted turn"])],
            role="user",
            max_tokens=128,
            audio_duration_seconds=12.5,
        )
        request_info = RequestInfo(
            model_endpoint=inference_client.model_endpoint,
            turns=[turn],
            turn_index=0,
            credit_num=0,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="test-id",
            x_correlation_id="test-corr",
            conversation_id="test-conv",
        )
        record = RequestRecord(
            request_info=request_info,
            start_perf_ns=1000,
            timestamp_ns=1000,
            end_perf_ns=2000,
        )

        result = inference_client._finalize_request_record(
            record=record, request_info=request_info
        )

        assert result.request_info.max_tokens == 128
        assert result.request_info.audio_duration_seconds == 12.5

    @pytest.mark.parametrize(
        "strip,expected_payload_bytes",
        [
            param(True, None, id="strip_enabled_drops_payload_bytes"),
            param(False, b'{"model": "test-model"}', id="strip_disabled_retains_payload_bytes"),
        ],
    )  # fmt: skip
    def test_finalize_request_record_payload_bytes_retention(
        self, inference_client, strip, expected_payload_bytes
    ):
        """With strip_record_payload_bytes enabled the canonical payload bytes
        are dropped from the RecordContext after dispatch (memory optimization
        resolved by the worker's payload-retention auto-detection); disabled
        leaves them intact for downstream consumers.
        """
        assert inference_client.strip_record_payload_bytes is False
        inference_client.strip_record_payload_bytes = strip
        request_info = RequestInfo(
            model_endpoint=inference_client.model_endpoint,
            turns=[Turn(texts=[Text(contents=["payload turn"])], role="user")],
            turn_index=0,
            credit_num=0,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="test-id",
            x_correlation_id="test-corr",
            conversation_id="test-conv",
            payload_bytes=b'{"model": "test-model"}',
        )
        record = RequestRecord(
            request_info=request_info,
            start_perf_ns=1000,
            timestamp_ns=1000,
            end_perf_ns=2000,
        )

        result = inference_client._finalize_request_record(
            record=record, request_info=request_info
        )

        assert result.request_info.payload_bytes == expected_payload_bytes

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


class TestInferenceClientSessionRouting:
    """Session-routing plugins wired through the InferenceClient chokepoint.

    The endpoint/transport plugins are mocked as before; the session_routing
    protocol resolves to the real routing classes so the chokepoint exercises
    genuine header/body transforms and the notify_session_end pass-through.
    """

    def _build_client(
        self,
        mock_http_transport_entry,
        *,
        session_routing: str | None,
        session_routing_opts: dict | None = None,
    ) -> InferenceClient:
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000/v1/test",
                session_routing=session_routing,
                session_routing_opts=session_routing_opts or {},
            ),
        )
        mock_transport = MagicMock()
        mock_endpoint = MagicMock()
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}
        mock_endpoint.format_payload.return_value = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }

        def mock_get_class(protocol, name):
            if protocol == "endpoint":
                return lambda **kwargs: mock_endpoint
            if protocol == "transport":
                return lambda **kwargs: mock_transport
            if protocol == "session_routing":
                return _ROUTING_CLASSES[name]
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
            client = InferenceClient(
                model_endpoint=model_endpoint, service_id="test-service-id"
            )
        client.transport.send_request = AsyncMock(return_value=RequestRecord())
        return client

    def _request_info(
        self,
        client: InferenceClient,
        *,
        x_correlation_id: str = "corr-1",
        parent_correlation_id: str | None = None,
        is_final_turn: bool = False,
    ) -> RequestInfo:
        return RequestInfo(
            model_endpoint=client.model_endpoint,
            turns=[Turn(role="user", texts=[Text(contents=["hello"])])],
            turn_index=0,
            credit_num=0,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="req-1",
            x_correlation_id=x_correlation_id,
            parent_correlation_id=parent_correlation_id,
            conversation_id="conv-template",
            is_final_turn=is_final_turn,
        )

    def _sent_payload(self, client: InferenceClient):
        payload = client.transport.send_request.call_args.kwargs["payload"]
        if isinstance(payload, bytes):
            return orjson.loads(payload)
        return payload

    @pytest.mark.asyncio
    async def test_dynamo_headers_mode_emits_headers_and_leaves_body(
        self, mock_http_transport_entry
    ):
        client = self._build_client(
            mock_http_transport_entry, session_routing="dynamo_headers"
        )
        request_info = self._request_info(
            client, parent_correlation_id="parent-corr", is_final_turn=False
        )

        await client._send_request_to_transport(request_info)

        assert request_info.endpoint_headers["X-Dynamo-Session-ID"] == "corr-1"
        assert (
            request_info.endpoint_headers["X-Dynamo-Parent-Session-ID"] == "parent-corr"
        )
        assert "nvext" not in self._sent_payload(client)

    @pytest.mark.asyncio
    async def test_dynamo_nvext_mode_binds_then_closes(self, mock_http_transport_entry):
        client = self._build_client(
            mock_http_transport_entry,
            session_routing="dynamo_nvext",
            session_routing_opts={"timeout_seconds": "123"},
        )

        non_final = self._request_info(client, is_final_turn=False)
        await client._send_request_to_transport(non_final)
        assert self._sent_payload(client)["nvext"]["session_control"] == {
            "session_id": "corr-1",
            "action": "bind",
            "timeout": 123,
        }

        final = self._request_info(client, is_final_turn=True)
        await client._send_request_to_transport(final)
        assert self._sent_payload(client)["nvext"]["session_control"] == {
            "session_id": "corr-1",
            "action": "close",
        }

    @pytest.mark.asyncio
    async def test_session_id_header_custom_name(self, mock_http_transport_entry):
        client = self._build_client(
            mock_http_transport_entry,
            session_routing="session_id_header",
            session_routing_opts={"header_name": "X-Affinity"},
        )
        request_info = self._request_info(client)

        await client._send_request_to_transport(request_info)

        assert request_info.endpoint_headers["X-Affinity"] == "corr-1"
        assert "nvext" not in self._sent_payload(client)

    @pytest.mark.asyncio
    async def test_routing_unset_no_headers_no_body_change(
        self, mock_http_transport_entry
    ):
        client = self._build_client(mock_http_transport_entry, session_routing=None)
        assert client._routing is None
        request_info = self._request_info(client, parent_correlation_id="parent-corr")

        await client._send_request_to_transport(request_info)

        payload = self._sent_payload(client)
        assert "nvext" not in payload
        assert "X-Dynamo-Session-ID" not in request_info.endpoint_headers
        assert "X-Dynamo-Parent-Session-ID" not in request_info.endpoint_headers

    @pytest.mark.asyncio
    async def test_payload_bytes_with_mutating_plugin_yields_error_record(
        self, mock_http_transport_entry
    ):
        client = self._build_client(
            mock_http_transport_entry, session_routing="dynamo_nvext"
        )
        request_info = self._request_info(client)
        request_info.payload_bytes = b'{"a":1}'

        record = await client.send_request(request_info)

        assert record.error is not None
        assert "PAYLOAD_BYTES" in record.error.message

    @pytest.mark.asyncio
    async def test_payload_bytes_with_header_plugin_gets_headers(
        self, mock_http_transport_entry
    ):
        client = self._build_client(
            mock_http_transport_entry, session_routing="dynamo_headers"
        )
        request_info = self._request_info(client, parent_correlation_id="parent-corr")
        request_info.payload_bytes = b'{"a":1}'

        await client._send_request_to_transport(request_info)

        assert request_info.endpoint_headers["X-Dynamo-Session-ID"] == "corr-1"
        assert (
            request_info.endpoint_headers["X-Dynamo-Parent-Session-ID"] == "parent-corr"
        )
        # The verbatim bytes are forwarded to the transport untouched.
        assert client.transport.send_request.call_args.kwargs["payload"] == b'{"a":1}'

    @pytest.mark.asyncio
    async def test_notify_session_end_reaches_plugin(self, mock_http_transport_entry):
        client = self._build_client(
            mock_http_transport_entry, session_routing="dynamo_headers"
        )
        client._routing.on_session_end = MagicMock()

        # Pass-through must not dedupe: idempotency is the plugin's job.
        client.notify_session_end("corr-1")
        client.notify_session_end("corr-1")

        assert client._routing.on_session_end.call_count == 2
        client._routing.on_session_end.assert_called_with("corr-1")

    def test_notify_session_end_noop_when_routing_unset(
        self, mock_http_transport_entry
    ):
        client = self._build_client(mock_http_transport_entry, session_routing=None)
        # No routing plugin: the hook is a safe no-op (never raises).
        client.notify_session_end("corr-1")

    def test_notify_session_end_swallows_plugin_error_and_warns(
        self, mock_http_transport_entry
    ):
        """A raising on_session_end must NOT propagate (core eviction must
        proceed); the failure is logged with the plugin + session named."""
        client = self._build_client(
            mock_http_transport_entry, session_routing="dynamo_headers"
        )
        client._routing.on_session_end = MagicMock(side_effect=RuntimeError("boom"))

        with patch.object(client, "warning") as warn:
            client.notify_session_end("corr-err")

        client._routing.on_session_end.assert_called_once_with("corr-err")
        warn.assert_called_once()
        msg = warn.call_args.args[0]
        assert "dynamo_headers" in msg and "corr-err" in msg

    @pytest.mark.asyncio
    async def test_raising_headers_produces_plugin_attributed_error_record(
        self, mock_http_transport_entry
    ):
        """A plugin exception in headers() surfaces as an error record naming the plugin."""
        client = self._build_client(
            mock_http_transport_entry, session_routing="dynamo_headers"
        )
        client._routing.headers = MagicMock(
            side_effect=RuntimeError("bad header build")
        )
        request_info = self._request_info(client)

        record = await client._send_request_internal(request_info)

        assert record.error is not None
        assert "dynamo_headers" in record.error.message
        assert "headers()" in record.error.message
        client.transport.send_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_raising_transform_body_produces_plugin_attributed_error_record(
        self, mock_http_transport_entry
    ):
        """A plugin exception in transform_body() is attributed to the plugin."""
        client = self._build_client(
            mock_http_transport_entry, session_routing="dynamo_nvext"
        )
        client._routing.transform_body = MagicMock(
            side_effect=RuntimeError("bad body transform")
        )
        request_info = self._request_info(client)

        record = await client._send_request_internal(request_info)

        assert record.error is not None
        assert "dynamo_nvext" in record.error.message
        assert "transform_body()" in record.error.message
