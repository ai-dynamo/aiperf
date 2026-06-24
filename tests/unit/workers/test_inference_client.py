# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
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

    @pytest.mark.asyncio
    async def test_send_request_allows_empty_turns_with_payload_bytes(
        self, inference_client
    ):
        """Empty turns must be accepted when payload_bytes provides the pre-built body."""
        request_info = RequestInfo(
            model_endpoint=inference_client.model_endpoint,
            turns=[],
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="test-id",
            x_correlation_id="test-corr",
            conversation_id="test-conv",
            payload_bytes=b'{"model":"test","messages":[]}',
        )

        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord(request_info=request_info)
        )

        record = await inference_client.send_request(request_info)
        assert record is not None

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

        result = inference_client._enrich_request_record(
            record=record, request_info=request_info
        )

        assert result.model_name == "standalone-model"

    @pytest.mark.asyncio
    async def test_send_request_uses_payload_bytes_when_set(
        self, inference_client, sample_request_info, sample_request_record
    ):
        """Test that payload_bytes bypasses endpoint.format_payload."""
        request_info = sample_request_info
        request_info.payload_bytes = (
            b'{"messages": [{"role": "user", "content": "raw"}]}'
        )

        inference_client.transport.send_request = AsyncMock(
            return_value=sample_request_record
        )

        await inference_client.send_request(request_info)

        # format_payload should NOT be called when payload_bytes is set
        inference_client.endpoint.format_payload.assert_not_called()
        call_args = inference_client.transport.send_request.call_args
        assert call_args.kwargs["payload"] == request_info.payload_bytes

    @pytest.mark.asyncio
    async def test_send_request_uses_raw_payload_from_turn(
        self, inference_client, sample_request_info, sample_request_record
    ):
        """Test that raw_payload on turn bypasses endpoint.format_payload."""
        import orjson

        from aiperf.common.models import Text, Turn

        raw = {"messages": [{"role": "user", "content": "raw turn"}], "model": "x"}
        request_info = sample_request_info
        request_info.turns = [
            Turn(role="user", raw_payload=raw, texts=[Text(contents=["x"])])
        ]
        request_info.turn_index = 0
        # ``sample_request_info`` pre-populates ``payload_bytes`` for ISL
        # tests; clear it here to exercise the raw_payload-on-turn branch
        # of ``_send_request_to_transport``.
        request_info.payload_bytes = None

        inference_client.transport.send_request = AsyncMock(
            return_value=sample_request_record
        )

        await inference_client.send_request(request_info)

        inference_client.endpoint.format_payload.assert_not_called()
        call_args = inference_client.transport.send_request.call_args
        # ``inference_client`` canonicalises the dict into bytes before
        # handing it to the transport so the record-processor replay path
        # has a stable ``request_info.payload_bytes`` to work from.
        assert call_args.kwargs["payload"] == orjson.dumps(raw)
        assert request_info.payload_bytes == orjson.dumps(raw)

    @pytest.mark.asyncio
    async def test_enrich_handles_empty_turns(
        self, inference_client, sample_request_info, sample_request_record
    ):
        """Test that _enrich_request_record handles turn_index >= len(turns)."""
        request_info = sample_request_info
        request_info.turns = []
        request_info.turn_index = 0

        record = sample_request_record
        enriched = inference_client._enrich_request_record(
            record=record, request_info=request_info
        )
        assert enriched.model_name == "test-model"

    def test_enrich_downcasts_to_slim_record_context(
        self, inference_client, model_endpoint
    ):
        """_enrich_request_record attaches a pure RecordContext, not the
        full RequestInfo. Pre-send-only surfaces (model_endpoint, turns,
        endpoint_headers, endpoint_params, drop_perf_ns, system_message,
        user_context_message) must not leak onto the record.

        This is the load-bearing invariant for the ZMQ slim-down: losing
        it silently re-inflates every record by ~500-900 bytes.
        """
        from aiperf.common.models.record_models import RecordContext

        turn = Turn(texts=[Text(contents=["x"])], role="user", model="test-model")
        request_info = RequestInfo(
            model_endpoint=model_endpoint,
            turns=[turn],
            turn_index=0,
            credit_num=7,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="rid",
            x_correlation_id="cid",
            conversation_id="conv",
            drop_perf_ns=12345,
            system_message="sys",
            user_context_message="uc",
            payload_bytes=b'{"model":"x","messages":[]}',
        )
        request_info.endpoint_headers = {"Authorization": "Bearer secret"}
        request_info.endpoint_params = {"api-version": "v1"}
        record = RequestRecord(
            request_info=request_info,
            start_perf_ns=1000,
            timestamp_ns=1000,
            end_perf_ns=2000,
        )

        enriched = inference_client._enrich_request_record(
            record=record, request_info=request_info
        )

        ctx = enriched.request_info
        assert ctx is not None
        # Slim: attached context is a pure RecordContext, not the RequestInfo
        # subclass. ``type`` equality (not isinstance) proves the down-cast.
        assert type(ctx) is RecordContext

        # Identity/routing scalars preserved.
        assert ctx.credit_num == 7
        assert ctx.conversation_id == "conv"
        assert ctx.turn_index == 0
        assert ctx.x_request_id == "rid"
        assert ctx.x_correlation_id == "cid"

        # Canonical wire body preserved.
        assert ctx.payload_bytes == b'{"model":"x","messages":[]}'

        # Pre-send-only surfaces stripped — accessing them on a pure
        # RecordContext raises AttributeError.
        for attr in (
            "model_endpoint",
            "turns",
            "endpoint_headers",
            "endpoint_params",
            "drop_perf_ns",
            "system_message",
            "user_context_message",
        ):
            assert not hasattr(ctx, attr), (
                f"RecordContext must not carry pre-send field {attr!r}"
            )

    def _enrich_with_payload(self, inference_client, model_endpoint):
        turn = Turn(texts=[Text(contents=["x"])], role="user", model="test-model")
        request_info = RequestInfo(
            model_endpoint=model_endpoint,
            turns=[turn],
            turn_index=0,
            credit_num=7,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="rid",
            x_correlation_id="cid",
            conversation_id="conv",
            payload_bytes=b'{"model":"x","messages":[{"role":"user","content":"x"}]}',
        )
        record = RequestRecord(
            request_info=request_info,
            start_perf_ns=1000,
            timestamp_ns=1000,
            end_perf_ns=2000,
        )
        enriched = inference_client._enrich_request_record(
            record=record, request_info=request_info
        )
        return enriched, request_info

    def test_enrich_strips_payload_bytes_when_flag_set(
        self, inference_client, model_endpoint
    ):
        """strip_record_payload_bytes=True omits huge request payloads from the
        record while leaving the source RequestInfo untouched."""
        inference_client.strip_record_payload_bytes = True
        enriched, request_info = self._enrich_with_payload(
            inference_client, model_endpoint
        )
        assert enriched.request_info is not None
        assert enriched.request_info.payload_bytes is None
        # Source RequestInfo is not mutated (transport already consumed it).
        assert request_info.payload_bytes is not None

    def test_enrich_keeps_payload_bytes_by_default(
        self, inference_client, model_endpoint
    ):
        """Default (flag False) preserves the canonical wire body on the record."""
        assert inference_client.strip_record_payload_bytes is False
        enriched, request_info = self._enrich_with_payload(
            inference_client, model_endpoint
        )
        assert enriched.request_info is not None
        assert enriched.request_info.payload_bytes == request_info.payload_bytes


class TestInferenceClientDynamoSessionControl:
    """Chokepoint injection of nvext.session_control for Dynamo routing.

    The verbatim PAYLOAD_BYTES path is refused against this feature at dataset
    load, so injection only ever runs on the structured (format_payload) body.
    """

    @pytest.fixture
    def model_endpoint(self):
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000/v1/test",
                use_dynamo_conv_aware_routing=True,
                dynamo_session_timeout_seconds=123,
            ),
        )

    @pytest.fixture
    def inference_client(self, model_endpoint, mock_http_transport_entry):
        mock_transport = MagicMock()
        mock_endpoint = MagicMock()
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}
        mock_endpoint.format_payload.return_value = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
        }

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

    def _request_info(
        self, inference_client, *, is_final_turn, x_correlation_id="corr-1"
    ):
        return RequestInfo(
            model_endpoint=inference_client.model_endpoint,
            turns=[Turn(role="user", texts=[Text(contents=["hi"])])],
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="rid",
            x_correlation_id=x_correlation_id,
            conversation_id="conv",
            is_final_turn=is_final_turn,
        )

    async def _sent_payload(self, inference_client, request_info):
        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord(request_info=request_info)
        )
        await inference_client.send_request(request_info)
        return orjson.loads(
            inference_client.transport.send_request.call_args.kwargs["payload"]
        )

    @pytest.mark.asyncio
    async def test_non_final_turn_binds_with_x_correlation_id_and_timeout(
        self, inference_client
    ):
        payload = await self._sent_payload(
            inference_client,
            self._request_info(inference_client, is_final_turn=False),
        )
        assert payload["nvext"]["session_control"] == {
            "session_id": "corr-1",
            "action": "bind",
            "timeout": 123,
        }
        # Endpoint-built fields are preserved.
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_final_turn_closes_session(self, inference_client):
        payload = await self._sent_payload(
            inference_client,
            self._request_info(inference_client, is_final_turn=True),
        )
        assert payload["nvext"]["session_control"] == {
            "session_id": "corr-1",
            "action": "close",
        }

    @pytest.mark.asyncio
    async def test_disabled_leaves_payload_untouched(self, inference_client):
        inference_client.model_endpoint.endpoint.use_dynamo_conv_aware_routing = False
        payload = await self._sent_payload(
            inference_client,
            self._request_info(inference_client, is_final_turn=False),
        )
        assert "nvext" not in payload


class TestInferenceClientLegacySessionControl:
    """Legacy (v1.2.x) open/close lifecycle, with the agentic-replay edge case.

    The critical property: 'open' fires on the FIRST request the worker sends
    for a session, tracked per-worker -- NOT on turn_index 0. Agentic replay
    warms at k_i and profiles from k_i+1, so the first request a worker sees for
    a session carries a NON-ZERO turn_index; a turn_index==0 gate would never
    emit 'open' for those sessions.
    """

    @pytest.fixture
    def model_endpoint(self):
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000/v1/test",
                use_dynamo_conv_aware_routing=True,
                use_legacy_dynamo_session_control=True,
                dynamo_session_timeout_seconds=123,
            ),
        )

    @pytest.fixture
    def inference_client(self, model_endpoint, mock_http_transport_entry):
        mock_transport = MagicMock()
        mock_endpoint = MagicMock()
        mock_endpoint.get_endpoint_headers.return_value = {}
        mock_endpoint.get_endpoint_params.return_value = {}
        mock_endpoint.format_payload.return_value = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
        }

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

    async def _sent_sc(
        self, inference_client, *, turn_index, is_final_turn, x_correlation_id="corr-1"
    ):
        request_info = RequestInfo(
            model_endpoint=inference_client.model_endpoint,
            turns=[Turn(role="user", texts=[Text(contents=["hi"])])],
            turn_index=turn_index,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="rid",
            x_correlation_id=x_correlation_id,
            conversation_id="conv",
            is_final_turn=is_final_turn,
        )
        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord(request_info=request_info)
        )
        await inference_client.send_request(request_info)
        payload = orjson.loads(
            inference_client.transport.send_request.call_args.kwargs["payload"]
        )
        return payload["nvext"]["session_control"]

    @pytest.mark.asyncio
    async def test_open_fires_on_first_request_with_nonzero_turn_index(
        self, inference_client
    ):
        """Agentx fix: the worker's first request for a session is the warmup
        turn at k_i (non-zero turn_index), and it must still emit 'open'."""
        sc = await self._sent_sc(inference_client, turn_index=5, is_final_turn=False)
        assert sc == {"session_id": "corr-1", "action": "open", "timeout": 123}

    @pytest.mark.asyncio
    async def test_open_once_then_session_id_only_then_close(self, inference_client):
        """Full lifecycle across the warmup->profiling boundary on one worker."""
        # warmup turn k_i: first request -> open
        warm = await self._sent_sc(inference_client, turn_index=5, is_final_turn=False)
        assert warm["action"] == "open"
        # profiling resume k_i+1: already opened -> session_id only, NO action
        mid = await self._sent_sc(inference_client, turn_index=6, is_final_turn=False)
        assert mid == {"session_id": "corr-1"}
        # another profiling turn: still session_id only
        mid2 = await self._sent_sc(inference_client, turn_index=7, is_final_turn=False)
        assert mid2 == {"session_id": "corr-1"}
        # final turn -> close
        final = await self._sent_sc(inference_client, turn_index=8, is_final_turn=True)
        assert final == {"session_id": "corr-1", "action": "close"}
        # 'open' emitted exactly once for the session.

    @pytest.mark.asyncio
    async def test_close_clears_tracking_state(self, inference_client):
        """The opened-sessions set is bounded: close drops the entry."""
        await self._sent_sc(inference_client, turn_index=5, is_final_turn=False)
        assert "corr-1" in inference_client._dynamo_opened_sessions
        await self._sent_sc(inference_client, turn_index=6, is_final_turn=True)
        assert "corr-1" not in inference_client._dynamo_opened_sessions

    @pytest.mark.asyncio
    async def test_each_session_opens_independently(self, inference_client):
        """Distinct sessions each get their own 'open'."""
        a = await self._sent_sc(
            inference_client, turn_index=2, is_final_turn=False, x_correlation_id="a"
        )
        b = await self._sent_sc(
            inference_client, turn_index=9, is_final_turn=False, x_correlation_id="b"
        )
        assert a["action"] == "open"
        assert b["action"] == "open"
        assert {"a", "b"} <= inference_client._dynamo_opened_sessions

    @pytest.mark.asyncio
    async def test_never_emits_bind(self, inference_client):
        """Legacy mode must never put 'bind' on the wire (v1.2.x rejects it)."""
        for ti in range(4):
            sc = await self._sent_sc(
                inference_client, turn_index=ti, is_final_turn=False
            )
            assert sc.get("action") != "bind"
