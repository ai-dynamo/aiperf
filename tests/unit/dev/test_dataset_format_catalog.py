# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dev.benchmarks.dataset_format_catalog import (
    EXCLUDED_FORMATS,
    FORMAT_PROFILES,
    SUPPORTED_FORMATS,
    SourceEnvelope,
    documented_skip_for,
    parity_fields_for,
    profile_for,
    registry_format_name,
)


def test_supported_catalog_has_twenty_two_families() -> None:
    assert len(SUPPORTED_FORMATS) == 22
    assert len(FORMAT_PROFILES) == 22


def test_burst_gpt_registry_alias() -> None:
    assert registry_format_name("burst_gpt_trace") == "burst_gpt"


def test_raw_payload_parity_omits_token_totals() -> None:
    assert "total_input_tokens" not in parity_fields_for("raw_payload")


def test_source_envelope_round_trip() -> None:
    envelope = SourceEnvelope(
        kind="inline_synthetic",
        inline={"marker": "__aiperf_synthetic", "synthetic_config": {"entries": 3}},
    )
    restored = SourceEnvelope.from_dict(envelope.to_dict())
    assert restored == envelope


def test_graph_and_accuracy_formats_are_explicitly_excluded() -> None:
    assert "accuracy" in EXCLUDED_FORMATS
    assert "dag_jsonl" in EXCLUDED_FORMATS


def test_streaming_pins_have_documented_skip_or_snapshot_cache() -> None:
    hf_conversation = profile_for("hf_conversation")
    hf_asr = profile_for("hf_asr")
    exgentic = profile_for("exgentic")
    exgentic_v2 = profile_for("exgentic_v2")
    assert hf_conversation is not None and hf_conversation.documented_skip
    assert hf_asr is not None and hf_asr.documented_skip
    assert exgentic is not None and exgentic.snapshot_cache
    assert exgentic_v2 is not None and exgentic_v2.snapshot_cache
    assert documented_skip_for("hf_conversation") == hf_conversation.documented_skip
    assert documented_skip_for("exgentic") is None
