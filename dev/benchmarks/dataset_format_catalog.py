# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative catalog for the Python/Rust dataset-load comparison harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import orjson

SourceKind = Literal["local_file", "inline_synthetic", "public_cached"]

DEFAULT_BENCHMARK_ROWS = 3

PARITY_SHAPE_FIELDS = ("row_count", "conversation_count", "turn_count")
PARITY_TOKEN_FIELD = "total_input_tokens"

EXCLUDED_FORMATS: dict[str, str] = {
    "accuracy": (
        "accuracy datasets are skipped because Python benchmark-plugin loading and "
        "Rust JSONL/static-accuracy loading are not equivalent pipelines"
    ),
    "dag_jsonl": (
        "graph formats are skipped because parse/lower pipelines do not produce the "
        "linear dataset measured here"
    ),
    "weka_trace": (
        "graph formats are skipped because parse/lower pipelines do not produce the "
        "linear dataset measured here"
    ),
    "dynamo_trace": (
        "graph formats are skipped because parse/lower pipelines do not produce the "
        "linear dataset measured here"
    ),
    "conditional_graph": (
        "graph formats are skipped because parse/lower pipelines do not produce the "
        "linear dataset measured here"
    ),
    "aiperf_trace": (
        "graph formats are skipped because parse/lower pipelines do not produce the "
        "linear dataset measured here"
    ),
}

UNVERIFIED_FORMAT_REASON = "format is not in the verified Python/Rust intersection"


@dataclass(frozen=True)
class FormatProfile:
    """One benchmarkable loader family and its cross-stack contract."""

    name: str
    rust_registry_name: str
    source_kind: SourceKind
    verified_options: dict[str, Any]
    parity_fields: tuple[str, ...]
    prepare_prompt_generator: bool
    public_pin: str | None = None
    public_aliases: tuple[str, ...] = ()
    requires_parquet: bool = False
    requires_async_python: bool = False
    opaque_token_counts: bool = False
    # A pinned public source whose files must be materialized on disk (via a
    # HuggingFace snapshot download) before the offline timed run, because the
    # source is a streaming dataset that HF cannot resolve from a partial cache.
    snapshot_cache: bool = False
    # When set, the family is supported in principle but cannot be benchmarked in
    # this harness's offline-cached model; the orchestrator records this reason as
    # a skip instead of attempting a prefetch or timed comparison.
    documented_skip: str | None = None


@dataclass(frozen=True)
class SourceEnvelope:
    """Normalized source description consumed by both adapters."""

    kind: SourceKind
    path: str | None = None
    inline: dict[str, Any] | None = None
    public: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.path is not None:
            payload["path"] = self.path
        if self.inline is not None:
            payload["inline"] = self.inline
        if self.public is not None:
            payload["public"] = self.public
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceEnvelope:
        kind = value.get("kind")
        if kind not in ("local_file", "inline_synthetic", "public_cached"):
            raise ValueError(f"unsupported source kind: {kind!r}")
        path = value.get("path")
        inline = value.get("inline")
        public = value.get("public")
        if not isinstance(path, (str, type(None))):
            raise ValueError("source.path must be a string when present")
        if inline is not None and not isinstance(inline, dict):
            raise ValueError("source.inline must be an object when present")
        if public is not None and not isinstance(public, dict):
            raise ValueError("source.public must be an object when present")
        return cls(
            kind=kind,
            path=path,
            inline=dict(inline) if inline is not None else None,
            public=dict(public) if public is not None else None,
        )

    def resolved_path(self) -> Path | None:
        if self.path is None:
            return None
        return Path(self.path)


@dataclass(frozen=True)
class FormatCase:
    """A format, source envelope, identity, and loader options to benchmark."""

    format: str
    fixture_id: str
    options: dict[str, object]
    source: SourceEnvelope

    @property
    def path(self) -> Path | None:
        return self.source.resolved_path()


def _shape_parity(
    *, include_tokens: bool = True, include_rows: bool = True
) -> tuple[str, ...]:
    fields: tuple[str, ...] = (
        PARITY_SHAPE_FIELDS
        if include_rows
        else (
            "conversation_count",
            "turn_count",
        )
    )
    if include_tokens:
        return (*fields, PARITY_TOKEN_FIELD)
    return fields


def _ranking_lengths_for_total_tokens(tokens_per_row: int) -> tuple[float, float]:
    """Split a total rankings ISL across query + 2 passages.

    The default synthetic-rankings shape is a `1:2:2` query-to-passages ratio
    (4 query tokens, 8 tokens per passage). Preserve that shape as closely as
    possible while keeping both passages the same length and making the three
    authored text chunks sum back to the requested total.
    """

    total_tokens = max(tokens_per_row, 3)
    desired_passage_tokens = round(total_tokens * 2 / 5)
    passage_tokens = max(1, min(desired_passage_tokens, (total_tokens - 1) // 2))
    query_tokens = total_tokens - (2 * passage_tokens)
    return float(query_tokens), float(passage_tokens)


def _shared_synthetic_inline(
    *, rankings: bool, entries: int, tokens_per_row: int | None = None
) -> dict[str, Any]:
    if rankings:
        query_tokens, passage_tokens = (
            (4.0, 8.0)
            if tokens_per_row is None
            else _ranking_lengths_for_total_tokens(tokens_per_row)
        )
        return {
            "marker": "__aiperf_synthetic_rankings",
            "synthetic_config": {
                "entries": entries,
                "turns": 1.0,
                "rankings": {
                    "passages": 2.0,
                    "passage_tokens": passage_tokens,
                    "query_tokens": query_tokens,
                },
            },
        }
    return {
        "marker": "__aiperf_synthetic",
        "synthetic_config": {
            "entries": entries,
            "turns": 1.0,
            "prompts": {
                "input_tokens": (
                    12.0 if tokens_per_row is None else float(max(tokens_per_row, 1))
                ),
                "output_tokens": 8.0,
                "batch_size": 1,
            },
        },
    }


FORMAT_PROFILES: dict[str, FormatProfile] = {
    "single_turn": FormatProfile(
        name="single_turn",
        rust_registry_name="single_turn",
        source_kind="local_file",
        verified_options={},
        parity_fields=_shape_parity(),
        prepare_prompt_generator=False,
    ),
    "multi_turn": FormatProfile(
        name="multi_turn",
        rust_registry_name="multi_turn",
        source_kind="local_file",
        verified_options={},
        parity_fields=_shape_parity(),
        prepare_prompt_generator=False,
    ),
    "raw_payload": FormatProfile(
        name="raw_payload",
        rust_registry_name="raw_payload",
        source_kind="local_file",
        verified_options={},
        parity_fields=PARITY_SHAPE_FIELDS,
        prepare_prompt_generator=False,
        opaque_token_counts=True,
    ),
    "inputs_json": FormatProfile(
        name="inputs_json",
        rust_registry_name="inputs_json",
        source_kind="local_file",
        verified_options={},
        parity_fields=PARITY_SHAPE_FIELDS,
        prepare_prompt_generator=False,
        opaque_token_counts=True,
    ),
    "random_pool": FormatProfile(
        name="random_pool",
        rust_registry_name="random_pool",
        source_kind="local_file",
        verified_options={},
        parity_fields=_shape_parity(),
        prepare_prompt_generator=False,
    ),
    "mooncake_trace": FormatProfile(
        name="mooncake_trace",
        rust_registry_name="mooncake_trace",
        source_kind="local_file",
        verified_options={},
        parity_fields=_shape_parity(),
        prepare_prompt_generator=True,
    ),
    "bailian_trace": FormatProfile(
        name="bailian_trace",
        rust_registry_name="bailian_trace",
        source_kind="local_file",
        verified_options={},
        parity_fields=_shape_parity(),
        prepare_prompt_generator=True,
    ),
    "burst_gpt_trace": FormatProfile(
        name="burst_gpt_trace",
        rust_registry_name="burst_gpt",
        source_kind="local_file",
        verified_options={},
        parity_fields=_shape_parity(),
        prepare_prompt_generator=True,
    ),
    "sagemaker_data_capture": FormatProfile(
        name="sagemaker_data_capture",
        rust_registry_name="sagemaker_data_capture",
        source_kind="local_file",
        verified_options={},
        parity_fields=_shape_parity(),
        prepare_prompt_generator=False,
    ),
    "baseten_trace": FormatProfile(
        name="baseten_trace",
        rust_registry_name="baseten_trace",
        source_kind="local_file",
        verified_options={},
        parity_fields=_shape_parity(),
        prepare_prompt_generator=False,
        requires_parquet=True,
    ),
    "speed_bench": FormatProfile(
        name="speed_bench",
        rust_registry_name="speed_bench",
        source_kind="local_file",
        verified_options={"category": "coding"},
        parity_fields=_shape_parity(),
        prepare_prompt_generator=False,
        public_aliases=(
            "speed_bench_coding",
            "speed_bench_math",
            "speed_bench_multilingual",
        ),
    ),
    "synthetic": FormatProfile(
        name="synthetic",
        rust_registry_name="synthetic",
        source_kind="inline_synthetic",
        verified_options={},
        parity_fields=_shape_parity(include_rows=False),
        prepare_prompt_generator=False,
    ),
    "synthetic_rankings": FormatProfile(
        name="synthetic_rankings",
        rust_registry_name="synthetic_rankings",
        source_kind="inline_synthetic",
        verified_options={},
        parity_fields=_shape_parity(include_rows=False),
        prepare_prompt_generator=False,
    ),
    "sharegpt": FormatProfile(
        name="sharegpt",
        rust_registry_name="sharegpt",
        source_kind="public_cached",
        verified_options={},
        parity_fields=_shape_parity(include_tokens=False),
        prepare_prompt_generator=False,
        public_pin="sharegpt",
        requires_async_python=True,
    ),
    "hf_instruction_response": FormatProfile(
        name="hf_instruction_response",
        rust_registry_name="hf_instruction_response",
        source_kind="public_cached",
        verified_options={"prompt_column": "question"},
        parity_fields=_shape_parity(include_tokens=False),
        prepare_prompt_generator=False,
        public_pin="spec_al_gsm8k",
        public_aliases=("aimo", "gsm8k", "spec_al_gsm8k"),
        requires_async_python=True,
    ),
    "hf_conversation": FormatProfile(
        name="hf_conversation",
        rust_registry_name="hf_conversation",
        source_kind="public_cached",
        verified_options={
            "conversation_column": "conversations",
            "message_content_key": "value",
        },
        parity_fields=_shape_parity(include_tokens=False),
        prepare_prompt_generator=False,
        public_pin="llava_onevision",
        requires_async_python=True,
        documented_skip=(
            "hf_conversation is pinned to the streaming lmms-lab/LLaVA-OneVision-Data "
            "dataset (~302 GB); it cannot be materialized for the harness's "
            "offline-cached timed run"
        ),
    ),
    "hf_asr": FormatProfile(
        name="hf_asr",
        rust_registry_name="hf_asr",
        source_kind="public_cached",
        verified_options={"audio_column": "audio"},
        parity_fields=_shape_parity(include_tokens=False),
        prepare_prompt_generator=False,
        public_pin="gigaspeech",
        public_aliases=("gigaspeech", "librispeech", "ami"),
        requires_async_python=True,
        documented_skip=(
            "hf_asr is pinned to the gated streaming speechcolab/gigaspeech dataset; "
            "offline caching requires interactive HuggingFace authentication that the "
            "harness cannot perform"
        ),
    ),
    "mt_bench": FormatProfile(
        name="mt_bench",
        rust_registry_name="mt_bench",
        source_kind="public_cached",
        verified_options={},
        parity_fields=_shape_parity(include_tokens=False),
        prepare_prompt_generator=False,
        public_pin="spec_al_mtbench",
        public_aliases=("mtbench", "spec_al_mtbench"),
        requires_async_python=True,
    ),
    "mmvu": FormatProfile(
        name="mmvu",
        rust_registry_name="mmvu",
        source_kind="public_cached",
        verified_options={"video_column": "video"},
        parity_fields=_shape_parity(include_tokens=False),
        prepare_prompt_generator=False,
        public_pin="mmvu",
        requires_async_python=True,
    ),
    "spec_bench": FormatProfile(
        name="spec_bench",
        rust_registry_name="spec_bench",
        source_kind="public_cached",
        verified_options={},
        parity_fields=_shape_parity(include_tokens=False),
        prepare_prompt_generator=False,
        public_pin="spec_bench",
        requires_async_python=True,
    ),
    "exgentic": FormatProfile(
        name="exgentic",
        rust_registry_name="exgentic",
        source_kind="public_cached",
        verified_options={"max_conversations": DEFAULT_BENCHMARK_ROWS},
        parity_fields=_shape_parity(include_tokens=False),
        prepare_prompt_generator=False,
        public_pin="exgentic",
        requires_async_python=True,
        snapshot_cache=True,
    ),
    "exgentic_v2": FormatProfile(
        name="exgentic_v2",
        rust_registry_name="exgentic_v2",
        source_kind="public_cached",
        verified_options={"max_conversations": DEFAULT_BENCHMARK_ROWS},
        parity_fields=_shape_parity(include_tokens=False),
        prepare_prompt_generator=False,
        public_pin="exgentic_v2",
        requires_async_python=True,
        snapshot_cache=True,
    ),
}

SUPPORTED_FORMATS: tuple[str, ...] = tuple(FORMAT_PROFILES)


def profile_for(format_name: str) -> FormatProfile | None:
    return FORMAT_PROFILES.get(format_name)


def unsupported_format_reason(format_name: str) -> str:
    if format_name in EXCLUDED_FORMATS:
        return EXCLUDED_FORMATS[format_name]
    return UNVERIFIED_FORMAT_REASON


def documented_skip_for(format_name: str) -> str | None:
    """Return the documented-skip reason for a supported-but-unrunnable family."""
    profile = profile_for(format_name)
    if profile is None:
        return None
    return profile.documented_skip


def parity_fields_for(format_name: str) -> tuple[str, ...]:
    profile = profile_for(format_name)
    if profile is None:
        return (*PARITY_SHAPE_FIELDS, PARITY_TOKEN_FIELD)
    return profile.parity_fields


def registry_format_name(format_name: str) -> str:
    profile = profile_for(format_name)
    if profile is None:
        return format_name
    return profile.rust_registry_name


def shared_synthetic_inline(
    *, rankings: bool, entries: int, tokens_per_row: int | None = None
) -> dict[str, Any]:
    return _shared_synthetic_inline(
        rankings=rankings,
        entries=entries,
        tokens_per_row=tokens_per_row,
    )


def source_envelope_json(source: SourceEnvelope) -> str:
    return orjson.dumps(source.to_dict()).decode()
