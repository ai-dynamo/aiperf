# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that the parser resolves a spec-decode adapter via the plugin registry.

Exercises ``InferenceResultParser._extract_spec_decode_acceptance`` -- the
auto-detection seam that walks registered ``spec_decode_adapter`` plugins -- so
each adapter is reachable end-to-end through the registry, not just when
imported directly.

AIPerf never learns which engine it is pointed at, by design: the user should
not have to declare it. Every payload is therefore offered to every registered
adapter, and detection soundness rests entirely on ``can_adapt`` signatures
being disjoint. These tests pin that contract from both directions -- each
adapter must claim its own payload and defer on the other's -- and pin the
parser's behaviour when a signature collision happens anyway.
"""

from typing import Any

from aiperf.common.models import ParsedResponse, SpecDecodeAcceptanceRecord
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.records.inference_result_parser import InferenceResultParser
from aiperf.spec_decode.trtllm_adapter import TRTLLMSpecDecodeAdapter
from aiperf.spec_decode.vllm_adapter import VLLMSpecDecodeAdapter
from tests.harness import mock_plugin
from tests.unit.spec_decode.test_trtllm_adapter import (
    SUMMARY_PAYLOAD as TRTLLM_PAYLOAD,
)
from tests.unit.spec_decode.test_vllm_adapter import SUMMARY_PAYLOAD


def test_adapters_are_registered() -> None:
    assert (
        plugins.get_class(PluginType.SPEC_DECODE_ADAPTER, "vllm").__name__
        == "VLLMSpecDecodeAdapter"
    )
    assert (
        plugins.get_class(PluginType.SPEC_DECODE_ADAPTER, "tensorrt_llm").__name__
        == "TRTLLMSpecDecodeAdapter"
    )


class TestCrossClaim:
    """Neither adapter may claim the other engine's payload.

    The two payloads share ``acceptance_histogram``, ``num_spec_steps``, and
    ``num_spec_tokens``, so a signature built only from shared keys would match
    both. Because the shared field names parse, the wrong adapter would build a
    record and label it with the wrong engine -- no exception, no dropped
    record, just a run attributed to a server that never produced it.
    """

    def test_vllm_defers_on_trtllm_payload(self) -> None:
        responses = [ParsedResponse(perf_ns=1, spec_decode_stats=TRTLLM_PAYLOAD)]
        assert VLLMSpecDecodeAdapter.can_adapt(responses) is False

    def test_trtllm_defers_on_vllm_payload(self) -> None:
        responses = [ParsedResponse(perf_ns=1, spec_decode_stats=SUMMARY_PAYLOAD)]
        assert TRTLLMSpecDecodeAdapter.can_adapt(responses) is False

    def test_each_payload_resolves_to_its_own_engine(self) -> None:
        for payload, engine in (
            (SUMMARY_PAYLOAD, "vllm"),
            (TRTLLM_PAYLOAD, "tensorrt_llm"),
        ):
            responses = [
                ParsedResponse(perf_ns=1, spec_decode_stats=payload),
                ParsedResponse(perf_ns=2, usage={"completion_tokens": 7}),
            ]
            record = InferenceResultParser._extract_spec_decode_acceptance(responses)
            assert isinstance(record, SpecDecodeAcceptanceRecord)
            assert record.engine == engine
            assert record.completion_tokens == 7


class TestAmbiguityIsRejected:
    """Two adapters claiming one payload is a bug, and must not be resolved.

    Iteration order is ``plugins.yaml`` declaration order -- ``priority`` only
    resolves conflicts between plugins registering the same *name*, and never
    orders iteration. So a first-match walk would silently let YAML line order
    decide which engine a record is attributed to.
    """

    class _GreedyAdapter:
        """Claims anything, standing in for a future adapter with a loose signature."""

        @classmethod
        def can_adapt(cls, responses: list[ParsedResponse]) -> bool:
            return any(r.spec_decode_stats for r in responses)

        @classmethod
        def adapt(cls, responses: list[ParsedResponse]) -> None:
            raise AssertionError("ambiguous payload must never be adapted")

    def test_contested_payload_yields_no_record(self) -> None:
        responses = [
            ParsedResponse(perf_ns=1, spec_decode_stats=SUMMARY_PAYLOAD),
            ParsedResponse(perf_ns=2, usage={"completion_tokens": 7}),
        ]
        with mock_plugin(
            PluginType.SPEC_DECODE_ADAPTER, "greedy-stub", self._GreedyAdapter
        ):
            # Two adapters now match; ``adapt`` raises if either is chosen, so a
            # passing test also proves no adapter was picked arbitrarily.
            assert (
                InferenceResultParser._extract_spec_decode_acceptance(responses) is None
            )


class TestAdapterFailureIsIsolated:
    """One adapter raising must not fail the record or hide the others.

    ``can_adapt`` is a third-party plugin callback: the protocol asks for it to
    be cheap and side-effect free, but nothing stops an implementation raising.
    Without isolation the exception propagates out of the parser and the whole
    inference record fails, even though the response was perfectly usable and a
    later adapter could have read it.
    """

    class _ExplodingAdapter:
        @classmethod
        def can_adapt(cls, responses: list[ParsedResponse]) -> bool:
            raise RuntimeError("third-party adapter blew up")

        @classmethod
        def adapt(cls, responses: list[ParsedResponse]) -> None:
            raise AssertionError("must never be reached")

    def test_raising_adapter_does_not_fail_the_record(self) -> None:
        responses = [
            ParsedResponse(perf_ns=1, spec_decode_stats=SUMMARY_PAYLOAD),
            ParsedResponse(perf_ns=2, usage={"completion_tokens": 7}),
        ]
        with mock_plugin(
            PluginType.SPEC_DECODE_ADAPTER, "exploding-stub", self._ExplodingAdapter
        ):
            record = InferenceResultParser._extract_spec_decode_acceptance(responses)

        # The working adapter is still found despite the broken one raising.
        assert record is not None
        assert record.engine == "vllm"


class TestNoAdapterMatches:
    def test_unrecognized_engine_payload_yields_none(self) -> None:
        """A payload no adapter recognizes yields no record rather than a guess.

        Guards the auto-detection contract: an adapter must not greedily claim a
        foreign payload just because the raw slot is populated.
        """
        responses = [
            ParsedResponse(
                perf_ns=1,
                spec_decode_stats={
                    "spec_correct_drafts_histogram": {"0": 5},
                    "steps": 5,
                },
            )
        ]
        assert InferenceResultParser._extract_spec_decode_acceptance(responses) is None


class TestPayloadPresence:
    def test_returns_none_when_no_stats(self) -> None:
        responses = [
            ParsedResponse(perf_ns=1),
            ParsedResponse(perf_ns=2, usage={"completion_tokens": 7}),
        ]
        assert InferenceResultParser._extract_spec_decode_acceptance(responses) is None

    def test_returns_none_for_empty_responses(self) -> None:
        assert InferenceResultParser._extract_spec_decode_acceptance([]) is None

    def test_suppresses_multi_sequence_stats(self) -> None:
        """n > 1 streaming: more than one response carries stats, so the
        per-request record is suppressed rather than mixing one sequence's
        acceptance with the request-level completion_tokens."""
        responses = [
            ParsedResponse(perf_ns=1, spec_decode_stats=SUMMARY_PAYLOAD),
            ParsedResponse(perf_ns=2, spec_decode_stats=SUMMARY_PAYLOAD),
            ParsedResponse(perf_ns=3, usage={"completion_tokens": 10}),
        ]
        assert InferenceResultParser._extract_spec_decode_acceptance(responses) is None

    def test_treats_empty_dict_payload_as_absent(self) -> None:
        """An empty ``{}`` payload is counted by truthiness (like the adapter), so
        a real payload beside it still yields a record instead of tripping the
        n > 1 guard."""
        responses: list[ParsedResponse] = [
            ParsedResponse(perf_ns=1, spec_decode_stats={}),
            ParsedResponse(perf_ns=2, spec_decode_stats=SUMMARY_PAYLOAD),
            ParsedResponse(perf_ns=3, usage={"completion_tokens": 5}),
        ]
        record = InferenceResultParser._extract_spec_decode_acceptance(responses)
        assert record is not None
        assert record.engine == "vllm"


def test_signature_keys_are_disjoint() -> None:
    """Mechanical guard on the property the whole design rests on.

    Fails the moment someone adds a key to one adapter's signature that the
    other engine also emits, instead of waiting for a mislabelled benchmark to
    surface it.
    """
    from aiperf.spec_decode.trtllm_adapter import _TRTLLM_SIGNATURE_KEYS
    from aiperf.spec_decode.vllm_adapter import _VLLM_SIGNATURE_KEYS

    vllm_payload: dict[str, Any] = SUMMARY_PAYLOAD
    trtllm_payload: dict[str, Any] = TRTLLM_PAYLOAD

    assert not all(key in trtllm_payload for key in _VLLM_SIGNATURE_KEYS)
    assert not all(key in vllm_payload for key in _TRTLLM_SIGNATURE_KEYS)
