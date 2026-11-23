# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The mock server's spec-decode payloads must be consumable by the adapters.

Every other test in this directory feeds the adapters hand-written payloads,
which only ever proves the adapters agree with our beliefs about the wire
format. This one closes the loop the other direction: it takes what the mock
server actually generates and runs it through the real adapter and the real
record validators, so a payload that could not produce a record fails here
rather than during a benchmark.

It also pins the mock's own contract -- that its acceptance numbers reconcile
with the ``completion_tokens`` it reports beside them, and that the record's
three identities hold by construction rather than by luck.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from aiperf_mock_server.config import server_config
from aiperf_mock_server.utils import attach_spec_decode, build_spec_decode_payload
from pytest import param

from aiperf.common.models import ParsedResponse, SpecDecodeAcceptanceRecord
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.spec_decode.trtllm_adapter import TRTLLMSpecDecodeAdapter
from aiperf.spec_decode.vllm_adapter import VLLMSpecDecodeAdapter


@pytest.fixture
def spec_decode_config():
    """Enable mock spec-decode emission, restoring prior settings afterwards."""
    saved = {
        key: getattr(server_config, key)
        for key in (
            "spec_decode_enabled",
            "spec_decode_flavor",
            "spec_decode_num_spec_tokens",
            "spec_decode_acceptance_rate",
        )
    }

    def configure(
        *, flavor: str, num_spec_tokens: int | None = 3, acceptance_rate: float = 0.5
    ) -> None:
        server_config.spec_decode_enabled = True
        server_config.spec_decode_flavor = flavor
        server_config.spec_decode_num_spec_tokens = num_spec_tokens
        server_config.spec_decode_acceptance_rate = acceptance_rate

    yield configure

    for key, value in saved.items():
        setattr(server_config, key, value)


def _ctx(completion_tokens: int) -> Any:
    return SimpleNamespace(usage={"completion_tokens": completion_tokens})


def _assert_identities(record: SpecDecodeAcceptanceRecord) -> None:
    """The invariants the record's validators enforce, restated explicitly.

    Constructing the record already checks these, so this documents what the
    mock is required to satisfy rather than adding coverage.
    """
    assert sum(record.acceptance_histogram.values()) == record.num_spec_steps
    assert (
        sum(j * c for j, c in record.acceptance_histogram.items())
        == record.num_accepted_draft_tokens
    )
    assert record.num_accepted_draft_tokens <= record.num_draft_tokens


class TestPayloadsProduceRecords:
    @pytest.mark.parametrize(
        "flavor, adapter, engine",
        [
            param("vllm", VLLMSpecDecodeAdapter, "vllm", id="vllm"),
            param("trtllm", TRTLLMSpecDecodeAdapter, "tensorrt_llm", id="trtllm"),
        ],
    )  # fmt: skip
    @pytest.mark.parametrize(
        "completion_tokens", [param(1, id="single_token"), param(100, id="typical")]
    )  # fmt: skip
    def test_adapter_builds_a_valid_record(
        self, spec_decode_config, flavor, adapter, engine, completion_tokens
    ) -> None:
        spec_decode_config(flavor=flavor)
        payload = build_spec_decode_payload(_ctx(completion_tokens))
        assert payload is not None

        responses = [ParsedResponse(perf_ns=1, spec_decode_stats=payload)]
        assert adapter.can_adapt(responses) is True
        record = adapter.adapt(responses)

        assert isinstance(record, SpecDecodeAcceptanceRecord)
        assert record.engine == engine
        _assert_identities(record)

    @pytest.mark.parametrize(
        "rate", [param(0.0, id="all_rejected"), param(1.0, id="all_accepted")]
    )  # fmt: skip
    def test_boundary_acceptance_rates(self, spec_decode_config, rate) -> None:
        """0.0 and 1.0 are the cases most likely to produce a degenerate payload."""
        spec_decode_config(flavor="vllm", acceptance_rate=rate)
        payload = build_spec_decode_payload(_ctx(100))
        record = VLLMSpecDecodeAdapter.adapt(
            [ParsedResponse(perf_ns=1, spec_decode_stats=payload)]
        )
        assert record is not None
        _assert_identities(record)
        assert record.draft_acceptance_rate == rate

    @pytest.mark.parametrize(
        "completion_tokens",
        [param(n, id=f"ct_{n}") for n in (1, 2, 3, 5, 17, 100)],
    )  # fmt: skip
    def test_acceptance_reconciles_with_usage(
        self, spec_decode_config, completion_tokens
    ) -> None:
        """Emitted acceptance must account for exactly the tokens usage reports.

        Each verify step emits one always-accepted bonus token plus its accepted
        drafts, so ``num_spec_steps + num_accepted == completion_tokens`` is a
        physical identity. The record's own validators do not cross-check against
        usage, so without this a payload can satisfy every identity while still
        claiming more emitted tokens than the response contained -- which it did
        for ``completion_tokens <= 2``.
        """
        spec_decode_config(flavor="vllm")
        payload = build_spec_decode_payload(_ctx(completion_tokens))
        assert (
            payload["num_spec_steps"] + payload["num_accepted_draft_tokens"]
            == completion_tokens
        )
        assert payload["num_accepted_draft_tokens"] <= payload["num_draft_tokens"]

    def test_adaptive_bound_round_trips(self, spec_decode_config) -> None:
        spec_decode_config(flavor="trtllm", num_spec_tokens=None)
        payload = build_spec_decode_payload(_ctx(100))
        assert payload["num_spec_tokens"] is None

        record = TRTLLMSpecDecodeAdapter.adapt(
            [ParsedResponse(perf_ns=1, spec_decode_stats=payload)]
        )
        assert record is not None
        assert record.num_spec_tokens is None
        _assert_identities(record)


class TestPlacementMatchesTheEngine:
    """``attach_spec_decode`` must put the payload where the capture layer looks."""

    @pytest.mark.parametrize(
        "flavor", [param("vllm", id="vllm"), param("trtllm", id="trtllm")]
    )  # fmt: skip
    def test_capture_layer_finds_the_payload(self, spec_decode_config, flavor) -> None:
        spec_decode_config(flavor=flavor)
        payload = build_spec_decode_payload(_ctx(100))
        response = attach_spec_decode(
            {
                "object": "chat.completion",
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 100},
            },
            payload,
        )

        assert BaseEndpoint.extract_spec_decode_stats(response) == payload

    def test_disabled_adds_no_key(self, spec_decode_config) -> None:
        """Off by default, responses must be untouched -- no null-filled key."""
        server_config.spec_decode_enabled = False
        response = {"object": "chat.completion", "choices": [{"text": "hi"}]}
        before = dict(response)
        assert (
            attach_spec_decode(response, build_spec_decode_payload(_ctx(100))) == before
        )
        assert "metrics" not in response
        assert "speculative_decoding" not in response["choices"][0]


class TestStreamingEndToEnd:
    """Each mock streaming path, through the real parser and adapter.

    The tests above call the payload helpers directly, so they cannot see which
    chunk a generator attaches the payload to. These drive the mock's streaming
    routes over HTTP, parse every chunk with AIPerf's own endpoint, and run the
    result through adapter auto-detection -- so a payload attached to the wrong
    chunk, attached twice, or dropped fails here.
    """

    @staticmethod
    def _stream(route: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        import orjson
        from aiperf_mock_server.app import asgi_app
        from fastapi.testclient import TestClient

        with TestClient(asgi_app) as client:
            resp = client.post(route, json=body)
        assert resp.status_code == 200
        return [
            orjson.loads(line[len("data: ") :])
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]

    @pytest.mark.parametrize(
        "flavor, engine", [("vllm", "vllm"), ("trtllm", "tensorrt_llm")]
    )
    @pytest.mark.parametrize(
        "route, body_key, endpoint_name",
        [
            param("/v1/chat/completions", "messages", "chat", id="chat"),
            param("/v1/completions", "prompt", "completions", id="completions"),
        ],
    )
    def test_streamed_payload_yields_a_record(
        self,
        spec_decode_config,
        monkeypatch,
        flavor,
        engine,
        route,
        body_key,
        endpoint_name,
    ) -> None:
        from aiperf_mock_server import utils as mock_utils

        from aiperf.endpoints.openai_chat import ChatEndpoint
        from aiperf.endpoints.openai_completions import CompletionsEndpoint
        from aiperf.plugin.enums import EndpointType
        from aiperf.records.inference_result_parser import InferenceResultParser
        from tests.unit.endpoints.test_spec_decode_capture import (
            _make_endpoint,
            _mock_response,
        )

        monkeypatch.setattr(mock_utils.server_config, "ttft", 0.0)
        monkeypatch.setattr(mock_utils.server_config, "itl", 0.0)
        spec_decode_config(flavor=flavor)
        prompt = (
            [{"role": "user", "content": "Write a few sentences."}]
            if body_key == "messages"
            else "Write a few sentences."
        )
        chunks = self._stream(
            route,
            {
                "model": "m",
                body_key: prompt,
                "max_tokens": 24,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        endpoint = (
            _make_endpoint(EndpointType.CHAT, ChatEndpoint)
            if endpoint_name == "chat"
            else _make_endpoint(EndpointType.COMPLETIONS, CompletionsEndpoint)
        )
        parsed = [
            p
            for p in (endpoint.parse_response(_mock_response(c)) for c in chunks)
            if p is not None
        ]

        record = InferenceResultParser._extract_spec_decode_acceptance(parsed)

        assert record is not None
        assert record.engine == engine
        streamed_usage = [c["usage"] for c in chunks if c.get("usage")]
        assert record.completion_tokens == streamed_usage[-1]["completion_tokens"]
        _assert_identities(record)

    @pytest.mark.parametrize(
        "flavor, engine", [("vllm", "vllm"), ("trtllm", "tensorrt_llm")]
    )
    def test_reasoning_only_stream_still_carries_the_payload(
        self, spec_decode_config, monkeypatch, flavor, engine
    ) -> None:
        """A reasoning model can spend every token on reasoning.

        The content loop then never runs, so a payload attached only to the
        last content chunk is never sent. It must ride the last chunk sent.
        """
        from aiperf_mock_server import utils as mock_utils

        from aiperf.endpoints.openai_chat import ChatEndpoint
        from aiperf.plugin.enums import EndpointType
        from aiperf.records.inference_result_parser import InferenceResultParser
        from tests.unit.endpoints.test_spec_decode_capture import (
            _make_endpoint,
            _mock_response,
        )

        monkeypatch.setattr(mock_utils.server_config, "ttft", 0.0)
        monkeypatch.setattr(mock_utils.server_config, "itl", 0.0)
        spec_decode_config(flavor=flavor)
        chunks = self._stream(
            "/v1/chat/completions",
            {
                # "qwen" marks a reasoning model; 24 tokens is below the
                # default reasoning budget, so no content token is produced.
                "model": "Qwen/Qwen3-0.6B",
                "messages": [{"role": "user", "content": "Write a few sentences."}],
                "max_tokens": 24,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert not any(
            c["choices"] and c["choices"][0].get("delta", {}).get("content")
            for c in chunks
        ), "precondition: the stream must contain reasoning only"
        endpoint = _make_endpoint(EndpointType.CHAT, ChatEndpoint)
        parsed = [
            p
            for p in (endpoint.parse_response(_mock_response(c)) for c in chunks)
            if p is not None
        ]

        record = InferenceResultParser._extract_spec_decode_acceptance(parsed)

        assert record is not None
        assert record.engine == engine
        streamed_usage = [c["usage"] for c in chunks if c.get("usage")]
        assert record.completion_tokens == streamed_usage[-1]["completion_tokens"]
