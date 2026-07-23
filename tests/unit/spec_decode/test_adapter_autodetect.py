# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that the parser resolves a spec-decode adapter via the plugin registry.

Exercises ``InferenceResultParser._extract_spec_decode_acceptance`` -- the
auto-detection seam that walks registered ``spec_decode_adapter`` plugins -- so
the vLLM adapter is reachable end-to-end through the registry, not just when
imported directly.
"""

from aiperf.common.models import ParsedResponse, SpecDecodeAcceptanceRecord
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.records.inference_result_parser import InferenceResultParser
from tests.unit.spec_decode.test_vllm_adapter import SUMMARY_PAYLOAD


def test_vllm_adapter_is_registered() -> None:
    """The vllm adapter resolves through the spec_decode_adapter category."""
    adapter = plugins.get_class(PluginType.SPEC_DECODE_ADAPTER, "vllm")
    assert adapter.__name__ == "VLLMSpecDecodeAdapter"


def test_extract_returns_record_when_stats_present() -> None:
    responses = [
        ParsedResponse(perf_ns=1, spec_decode_stats=SUMMARY_PAYLOAD),
        ParsedResponse(perf_ns=2, usage={"completion_tokens": 7}),
    ]
    record = InferenceResultParser._extract_spec_decode_acceptance(responses)

    assert isinstance(record, SpecDecodeAcceptanceRecord)
    assert record.engine == "vllm"
    assert record.completion_tokens == 7


def test_extract_returns_none_when_no_stats() -> None:
    """Records whose responses carry no spec-decode payload yield None."""
    responses = [
        ParsedResponse(perf_ns=1),
        ParsedResponse(perf_ns=2, usage={"completion_tokens": 7}),
    ]
    assert InferenceResultParser._extract_spec_decode_acceptance(responses) is None


def test_extract_returns_none_for_empty_responses() -> None:
    """No responses at all yields None."""
    assert InferenceResultParser._extract_spec_decode_acceptance([]) is None


def test_extract_returns_none_for_unrecognized_engine_payload() -> None:
    """A payload present but matching no registered adapter yields no record.

    Guards the auto-detection contract: the vLLM adapter must not greedily claim
    a foreign (e.g. future SGLang/TRT-LLM) payload just because the raw slot is
    populated. When another adapter is added, it -- not vLLM -- should win here.
    """
    responses = [
        ParsedResponse(
            perf_ns=1,
            spec_decode_stats={"spec_correct_drafts_histogram": {"0": 5}, "steps": 5},
        )
    ]
    assert InferenceResultParser._extract_spec_decode_acceptance(responses) is None


def test_extract_suppresses_multi_sequence_stats() -> None:
    """n > 1 streaming: more than one response carries stats, so the per-request
    record is suppressed rather than mixing one sequence's acceptance with the
    request-level completion_tokens."""
    responses = [
        ParsedResponse(perf_ns=1, spec_decode_stats=SUMMARY_PAYLOAD),
        ParsedResponse(perf_ns=2, spec_decode_stats=SUMMARY_PAYLOAD),
        ParsedResponse(perf_ns=3, usage={"completion_tokens": 10}),
    ]
    assert InferenceResultParser._extract_spec_decode_acceptance(responses) is None
