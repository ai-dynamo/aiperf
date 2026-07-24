# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin Python adapter for dataset load, composition, and tokenization timing.

See ``dev/benchmarks/README.md`` for the shared harness contract and the
benchmark-only tokenizer/chat-template/exact-ISL modes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aiperf.common import random_generator as rng
from aiperf.common.enums import CreditPhase, MediaType
from aiperf.common.models import Conversation, ModelEndpointInfo
from aiperf.common.tokenizer import Tokenizer
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer
from aiperf.dataset.composer.synthetic_rankings import SyntheticRankingsDatasetComposer
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.bailian_trace import BailianTraceDatasetLoader
from aiperf.dataset.loader.baseten_trace import BasetenTraceDatasetLoader
from aiperf.dataset.loader.burst_gpt import BurstGPTTraceDatasetLoader
from aiperf.dataset.loader.exgentic import ExgenticDatasetLoader
from aiperf.dataset.loader.exgentic_v2 import ExgenticV2DatasetLoader
from aiperf.dataset.loader.hf_asr import HFASRDatasetLoader
from aiperf.dataset.loader.hf_conversation import HFConversationDatasetLoader
from aiperf.dataset.loader.hf_instruction_response import (
    HFInstructionResponseDatasetLoader,
)
from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader
from aiperf.dataset.loader.mmvu import MMVUDatasetLoader
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.dataset.loader.mt_bench import MTBenchDatasetLoader
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader
from aiperf.dataset.loader.random_pool import RandomPoolDatasetLoader
from aiperf.dataset.loader.raw_payload import RawPayloadDatasetLoader
from aiperf.dataset.loader.sagemaker_data_capture import SageMakerDataCaptureLoader
from aiperf.dataset.loader.sharegpt import ShareGPTLoader
from aiperf.dataset.loader.single_turn import SingleTurnDatasetLoader
from aiperf.dataset.loader.spec_bench import SpecBenchLoader
from aiperf.dataset.loader.speed_bench import SpeedBenchLoader
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.payload_extraction import extract_inputs
from dev.benchmarks.dataset_format_catalog import SourceEnvelope, profile_for
from dev.benchmarks.dataset_public_cache import (
    _merge_options,
    _public_entry,
    effective_streaming,
)

FILE_LOADERS = {
    "single_turn": SingleTurnDatasetLoader,
    "multi_turn": MultiTurnDatasetLoader,
    "raw_payload": RawPayloadDatasetLoader,
    "inputs_json": InputsJsonPayloadLoader,
    "random_pool": RandomPoolDatasetLoader,
    "mooncake_trace": MooncakeTraceDatasetLoader,
    "bailian_trace": BailianTraceDatasetLoader,
    "burst_gpt_trace": BurstGPTTraceDatasetLoader,
    "sagemaker_data_capture": SageMakerDataCaptureLoader,
    "baseten_trace": BasetenTraceDatasetLoader,
    "speed_bench": SpeedBenchLoader,
}

_CHAT_PART_TYPES = {
    MediaType.TEXT: {"text"},
    MediaType.IMAGE: {"image_url"},
    MediaType.AUDIO: {"input_audio"},
    MediaType.VIDEO: {"video_url"},
}


@dataclass(frozen=True)
class Sample:
    """One adapter measurement using the shared comparison schema."""

    implementation: str
    format: str
    fixture_id: str
    row_count: int
    conversation_count: int
    turn_count: int
    total_input_tokens: int | None
    elapsed_ns: int
    error: str | None = None


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _benchmark_run(
    *,
    seed: int,
    model: str,
    entries: int = 3,
    synthetic_shape: dict[str, Any] | None = None,
) -> BenchmarkRun:
    dataset: dict[str, Any] = {
        "name": "adapter",
        "type": "synthetic",
        "entries": entries,
        "prompts": {"isl": 1, "osl": 1},
    }
    if synthetic_shape is not None:
        dataset = {
            "name": "adapter",
            "type": "synthetic",
            "entries": synthetic_shape.get("entries", entries),
            "prompts": {
                "isl": int(synthetic_shape.get("prompts", {}).get("input_tokens", 12)),
                "osl": int(synthetic_shape.get("prompts", {}).get("output_tokens", 8)),
            },
            "turns": int(synthetic_shape.get("turns", 1)),
        }
        if "rankings" in synthetic_shape:
            rankings = synthetic_shape["rankings"]
            dataset["rankings"] = {
                "passages": int(rankings.get("passages", 2)),
                "passage_tokens": int(rankings.get("passage_tokens", 8)),
                "query_tokens": int(rankings.get("query_tokens", 4)),
            }
    cfg = BenchmarkConfig.model_validate(
        {
            "models": [model],
            "endpoint": {
                "urls": ["http://localhost:8000/v1/chat/completions"],
                "wait_for_model_timeout": 0,
            },
            "datasets": [dataset],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": entries,
                    "concurrency": 1,
                }
            ],
            "runtime": {"ui": "simple"},
        }
    )
    return BenchmarkRun(
        benchmark_id="dataset-load-python",
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=seed,
        cli_command=None,
    )


def _count_loaded_rows(format_name: str, loaded: dict[str, list[Any]]) -> int:
    if format_name == "inputs_json":
        return sum(
            len(session.payloads)
            for sessions in loaded.values()
            for session in sessions
        )
    if format_name in {"synthetic", "synthetic_rankings"}:
        return sum(len(rows) for rows in loaded.values()) if loaded else 0
    return sum(len(rows) for rows in loaded.values())


def _public_row_limit(source: SourceEnvelope, options: dict[str, object]) -> int:
    """Resolve the loaded-row cap for a public source, matching the Rust adapter.

    Rust truncates loaded rows to ``identity.row_limit`` (falling back to a
    ``max_conversations`` option and finally 3); Python mirrors that ordering so
    both implementations load and compose the same number of rows.
    """
    public = source.public or {}
    identity = public.get("identity")
    if isinstance(identity, dict) and isinstance(identity.get("row_limit"), int):
        return int(identity["row_limit"])
    max_conversations = options.get("max_conversations")
    if isinstance(max_conversations, int):
        return max_conversations
    return 3


def _loaded_rows_view(loaded: Any) -> Any:
    """Return the underlying row collection for a loader's raw output."""
    if isinstance(loaded, dict) and "dataset" in loaded:
        return loaded["dataset"]
    return loaded


def _truncate_rows(rows: Any, limit: int) -> Any:
    """Truncate a list, HuggingFace ``Dataset``, or ``IterableDataset`` to ``limit``."""
    if isinstance(rows, list):
        return rows[:limit]
    select = getattr(rows, "select", None)
    if callable(select) and hasattr(rows, "__len__"):
        return select(range(min(limit, len(rows))))
    take = getattr(rows, "take", None)
    if callable(take):
        return take(limit)
    return rows


def _truncate_loaded(loaded: Any, limit: int) -> Any:
    """Cap a loader's raw output to ``limit`` rows before composition."""
    rows = _loaded_rows_view(loaded)
    truncated = _truncate_rows(rows, limit)
    if isinstance(loaded, dict) and "dataset" in loaded:
        capped = dict(loaded)
        capped["dataset"] = truncated
        return capped
    return truncated


def _public_row_count(loaded: Any) -> int:
    """Count loaded rows across list, ``{"dataset": ...}``, and iterable shapes."""
    rows = _loaded_rows_view(loaded)
    try:
        return len(rows)
    except TypeError:
        return sum(1 for _ in rows)


def _count_input_tokens(
    conversations: list[Conversation],
    tokenizer: Tokenizer,
    *,
    apply_chat_template: bool,
    chat_endpoint: ChatEndpoint | None,
) -> int:
    total = 0
    for conversation in conversations:
        for turn in conversation.turns:
            if turn.raw_payload is not None:
                extracted = extract_inputs(turn.raw_payload, _CHAT_PART_TYPES)
                total += _count_extracted_tokens(
                    extracted,
                    tokenizer,
                    apply_chat_template=apply_chat_template,
                )
                continue
            if turn.raw_messages is not None:
                payload: dict[str, object] = {"messages": turn.raw_messages}
                if turn.raw_tools is not None:
                    payload["tools"] = turn.raw_tools
                if apply_chat_template:
                    payload["messages"] = ChatEndpoint._format_messages(
                        SimpleNamespace(
                            system_message=conversation.system_message,
                            user_context_message=conversation.user_context_message,
                            credit_phase=CreditPhase.PROFILING,
                        ),
                        list(turn.raw_messages),
                    )
                extracted = extract_inputs(payload, _CHAT_PART_TYPES)
                total += _count_extracted_tokens(
                    extracted,
                    tokenizer,
                    apply_chat_template=apply_chat_template,
                )
                continue
            if apply_chat_template and chat_endpoint is not None:
                extracted = extract_inputs(
                    _chat_payload_for_turn(conversation, turn, chat_endpoint),
                    _CHAT_PART_TYPES,
                )
                total += _count_extracted_tokens(
                    extracted,
                    tokenizer,
                    apply_chat_template=True,
                )
                continue
            for text in turn.texts:
                for content in text.contents:
                    total += len(tokenizer.encode(content))
    return total


def _chat_template_token_count(
    tokenizer: Tokenizer, messages: list[dict[str, str]]
) -> int | None:
    inner = getattr(tokenizer, "_tokenizer", None)
    apply = getattr(inner, "apply_chat_template", None)
    if apply is None or not messages:
        return None
    if getattr(inner, "chat_template", "_unset") is None:
        return None
    try:
        tokens = apply(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception:
        return None
    if isinstance(tokens, list):
        return len(tokens)
    input_ids = getattr(tokens, "input_ids", None)
    if isinstance(input_ids, list):
        return len(input_ids)
    if isinstance(tokens, dict):
        mapped_ids = tokens.get("input_ids")
        if isinstance(mapped_ids, list):
            return len(mapped_ids)
    return None


def _count_extracted_tokens(
    extracted: Any,
    tokenizer: Tokenizer,
    *,
    apply_chat_template: bool,
) -> int:
    total = extracted.pretokenised_token_count
    if apply_chat_template and extracted.messages:
        templated = _chat_template_token_count(tokenizer, extracted.messages)
        if templated is not None:
            total += templated
            total += sum(len(tokenizer.encode(text)) for text in extracted.tool_texts)
            return total
    total += sum(len(tokenizer.encode(text)) for text in extracted.texts)
    return total


def _chat_payload_for_turn(
    conversation: Conversation,
    turn: Any,
    chat_endpoint: ChatEndpoint,
) -> dict[str, object]:
    rendered = chat_endpoint.build_messages([turn])
    payload: dict[str, object] = {
        "messages": ChatEndpoint._format_messages(
            SimpleNamespace(
                system_message=conversation.system_message,
                user_context_message=conversation.user_context_message,
                credit_phase=CreditPhase.PROFILING,
            ),
            rendered,
        )
    }
    if turn.raw_tools is not None:
        payload["tools"] = turn.raw_tools
    return payload


def _stored_input_tokens(conversations: list[Conversation]) -> int | None:
    total = 0
    for conversation in conversations:
        for turn in conversation.turns:
            value = getattr(turn, "input_tokens", None)
            if not isinstance(value, int):
                return None
            total += value
    return total


def _create_file_loader(
    format_name: str,
    path: Path,
    run: BenchmarkRun,
    tokenizer: Tokenizer,
    options: dict[str, object],
) -> Any:
    loader_class = FILE_LOADERS[format_name]
    kwargs: dict[str, object] = {}
    if format_name in {
        "mooncake_trace",
        "bailian_trace",
        "burst_gpt_trace",
        "sagemaker_data_capture",
        "baseten_trace",
    }:
        kwargs["prompt_generator"] = PromptGenerator(
            prompts=None,
            prefix_prompts=None,
            tokenizer=tokenizer,
        )
    if format_name == "bailian_trace":
        kwargs["default_block_size"] = 16
    if format_name == "random_pool":
        kwargs["num_conversations"] = 1
    if format_name == "speed_bench":
        kwargs["category"] = options.get("category")
        kwargs["multi_turn"] = True
    return loader_class(filename=path, run=run, **kwargs)


def _create_public_loader(
    format_name: str,
    source: SourceEnvelope,
    run: BenchmarkRun,
    tokenizer: Tokenizer,
    options: dict[str, object],
) -> Any:
    public = source.public or {}
    pin_key = public.get("pin_key")
    if not isinstance(pin_key, str):
        raise ValueError("public_cached source requires pin_key")
    entry = _public_entry(pin_key)
    merged = _merge_options(entry, options)
    entry_format = str(entry.get("format", format_name))
    profile = profile_for(format_name)
    if profile is None:
        raise ValueError(f"unsupported public format {format_name!r}")
    streaming = effective_streaming(profile, entry)
    hf_source = entry["source"]

    if entry_format == "sharegpt":
        return ShareGPTLoader(run=run, tokenizer=tokenizer)
    if entry_format == "spec_bench":
        return SpecBenchLoader(run=run, multi_turn=False)
    if entry_format == "hf_instruction_response":
        return HFInstructionResponseDatasetLoader(
            run=run,
            hf_dataset_name=hf_source["dataset"],
            hf_split=hf_source.get("split", "train"),
            hf_subset=hf_source.get("subset"),
            streaming=streaming,
            hf_revision=hf_source.get("revision"),
            prompt_column=str(merged["prompt_column"]),
            image_column=merged.get("image_column"),
            video_column=merged.get("video_column"),
            audio_column=merged.get("audio_column"),
            prompt_template=merged.get("prompt_template"),
        )
    if entry_format == "hf_conversation":
        return HFConversationDatasetLoader(
            run=run,
            hf_dataset_name=hf_source["dataset"],
            hf_split=hf_source.get("split", "train"),
            hf_subset=hf_source.get("subset"),
            streaming=streaming,
            hf_revision=hf_source.get("revision"),
            conversation_column=str(merged["conversation_column"]),
            message_content_key=str(merged.get("message_content_key", "value")),
            image_column=merged.get("image_column"),
            video_column=merged.get("video_column"),
        )
    if entry_format == "hf_asr":
        return HFASRDatasetLoader(
            run=run,
            hf_dataset_name=hf_source["dataset"],
            hf_split=hf_source.get("split", "train"),
            hf_subset=hf_source.get("subset"),
            streaming=streaming,
            hf_revision=hf_source.get("revision"),
            audio_column=str(merged["audio_column"]),
        )
    if entry_format == "mt_bench":
        return MTBenchDatasetLoader(
            run=run,
            hf_dataset_name=hf_source["dataset"],
            hf_split=hf_source.get("split", "train"),
            hf_subset=hf_source.get("subset"),
            streaming=streaming,
            hf_revision=hf_source.get("revision"),
        )
    if entry_format == "mmvu":
        return MMVUDatasetLoader(
            run=run,
            hf_dataset_name=hf_source["dataset"],
            hf_split=hf_source.get("split", "train"),
            hf_subset=hf_source.get("subset"),
            streaming=streaming,
            hf_revision=hf_source.get("revision"),
            video_column=str(merged["video_column"]),
        )
    if entry_format == "exgentic":
        return ExgenticDatasetLoader(
            run=run,
            hf_dataset_name=hf_source["dataset"],
            hf_split=hf_source.get("split", "train"),
            hf_subset=hf_source.get("subset"),
            streaming=streaming,
            hf_revision=hf_source.get("revision"),
        )
    if entry_format == "exgentic_v2":
        return ExgenticV2DatasetLoader(
            run=run,
            hf_dataset_name=hf_source["dataset"],
            hf_split=hf_source.get("split", "train"),
            hf_subset=hf_source.get("subset"),
            streaming=streaming,
            hf_revision=hf_source.get("revision"),
        )
    raise ValueError(f"unsupported public format {entry_format!r}")


async def _load_and_compose_async(
    format_name: str,
    *,
    path: Path | None,
    source: SourceEnvelope,
    options: dict[str, object],
    run: BenchmarkRun,
    tokenizer: Tokenizer,
    tokenizer_name: str,
    apply_chat_template: bool,
    chat_endpoint: ChatEndpoint | None,
    exact_isl: bool,
) -> tuple[dict[str, list[Any]] | None, list[Conversation], int | None]:
    profile = profile_for(format_name)
    if profile is None:
        raise ValueError(f"unsupported dataset format: {format_name}")

    if source.kind == "inline_synthetic":
        inline = source.inline or {}
        shape = inline.get("synthetic_config", {})
        synthetic_run = _benchmark_run(
            seed=run.random_seed,
            model=run.cfg.get_model_names()[0],
            entries=int(shape.get("entries", 3)),
            synthetic_shape=shape,
        )
        if format_name == "synthetic_rankings":
            composer = SyntheticRankingsDatasetComposer(
                run=synthetic_run, tokenizer=tokenizer
            )
        else:
            composer = SyntheticDatasetComposer(run=synthetic_run, tokenizer=tokenizer)
        conversations = composer.create_dataset()
        total_input_tokens = _total_input_tokens(
            format_name,
            {},
            conversations,
            tokenizer,
            tokenizer_name=tokenizer_name,
            apply_chat_template=apply_chat_template,
            chat_endpoint=chat_endpoint,
            exact_isl=exact_isl,
        )
        return None, conversations, total_input_tokens

    if source.kind == "public_cached":
        limit = _public_row_limit(source, options)
        loader = _create_public_loader(format_name, source, run, tokenizer, options)
        loaded = await loader.load_dataset()
        loaded = _truncate_loaded(loaded, limit)
        conversations = await loader.convert_to_conversations(loaded)
        conversations = conversations[:limit]
        row_count = _public_row_count(loaded)
        total_input_tokens = _total_input_tokens(
            format_name,
            {format_name: [object()] * row_count},
            conversations,
            tokenizer,
            tokenizer_name=tokenizer_name,
            apply_chat_template=apply_chat_template,
            chat_endpoint=chat_endpoint,
            exact_isl=exact_isl,
        )
        return {format_name: [object()] * row_count}, conversations, total_input_tokens

    if path is None:
        raise ValueError(f"{format_name} requires a local file path")
    loader = _create_file_loader(format_name, path, run, tokenizer, options)
    loaded = loader.load_dataset()
    conversations = loader.convert_to_conversations(loaded)
    return (
        loaded,
        conversations,
        _total_input_tokens(
            format_name,
            loaded,
            conversations,
            tokenizer,
            tokenizer_name=tokenizer_name,
            apply_chat_template=apply_chat_template,
            chat_endpoint=chat_endpoint,
            exact_isl=exact_isl,
        ),
    )


def _total_input_tokens(
    format_name: str,
    loaded: dict[str, list[Any]],
    conversations: list[Conversation],
    tokenizer: Tokenizer,
    *,
    tokenizer_name: str,
    apply_chat_template: bool,
    chat_endpoint: ChatEndpoint | None,
    exact_isl: bool,
) -> int | None:
    profile = profile_for(format_name)
    rich_token_counts = apply_chat_template or tokenizer_name != "builtin"
    if profile is not None and profile.opaque_token_counts and not rich_token_counts:
        return None
    if format_name in {"bailian_trace", "burst_gpt_trace"}:
        return sum(trace.input_length for traces in loaded.values() for trace in traces)
    if format_name == "baseten_trace":
        return sum(trace.input_tokens for traces in loaded.values() for trace in traces)
    if (
        profile is not None
        and profile.parity_fields
        == (
            "row_count",
            "conversation_count",
            "turn_count",
        )
        and not rich_token_counts
    ):
        return None
    if (
        not rich_token_counts
        and not exact_isl
        and format_name in {"synthetic", "synthetic_rankings"}
    ):
        stored = _stored_input_tokens(conversations)
        if stored is not None:
            return stored
    return _count_input_tokens(
        conversations,
        tokenizer,
        apply_chat_template=apply_chat_template,
        chat_endpoint=chat_endpoint,
    )


def run_sample(
    *,
    format_name: str,
    path: Path | None,
    source: SourceEnvelope,
    options: dict[str, object],
    fixture_id: str,
    seed: int,
    model: str,
    tokenizer_name: str = "builtin",
    apply_chat_template: bool = False,
    exact_isl: bool = False,
) -> Sample:
    """Measure one load, compose, and tokenize operation."""
    profile = profile_for(format_name)
    if profile is None:
        raise ValueError(f"unsupported dataset format: {format_name}")
    if options != profile.verified_options:
        raise ValueError(
            f"options {options!r} do not match verified mapping "
            f"{profile.verified_options!r}"
        )

    rng.reset()
    rng.init(seed)
    entry_count = int(options.get("max_conversations", 3))
    run = _benchmark_run(seed=seed, model=model, entries=entry_count)
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.encode("warm")
    chat_endpoint = (
        ChatEndpoint(ModelEndpointInfo.from_run(run)) if apply_chat_template else None
    )

    started_ns = time.perf_counter_ns()
    loaded, conversations, total_input_tokens = asyncio.run(
        _load_and_compose_async(
            format_name,
            path=path,
            source=source,
            options=options,
            run=run,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            apply_chat_template=apply_chat_template,
            chat_endpoint=chat_endpoint,
            exact_isl=exact_isl,
        )
    )
    elapsed_ns = time.perf_counter_ns() - started_ns

    if loaded is None:
        row_count = len(conversations)
    else:
        row_count = _count_loaded_rows(format_name, loaded)
    turn_count = sum(len(conversation.turns) for conversation in conversations)
    return Sample(
        implementation="python",
        format=format_name,
        fixture_id=fixture_id,
        row_count=row_count,
        conversation_count=len(conversations),
        turn_count=turn_count,
        total_input_tokens=total_input_tokens,
        elapsed_ns=elapsed_ns,
    )


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--format", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--options-json", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="builtin")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--exact-isl", action="store_true")
    return parser


def _parse_options(raw: str) -> dict[str, object]:
    options = json.loads(raw)
    if not isinstance(options, dict):
        raise ValueError("--options-json must decode to a JSON object")
    return options


def _parse_source(raw: str) -> SourceEnvelope:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--source-json must decode to a JSON object")
    return SourceEnvelope.from_dict(payload)


def _path_or_none(raw: str) -> Path | None:
    if not raw:
        return None
    return Path(raw)


def _error_sample(format_name: str, fixture_id: str, error: Exception) -> Sample:
    return Sample(
        implementation="python",
        format=format_name,
        fixture_id=fixture_id,
        row_count=0,
        conversation_count=0,
        turn_count=0,
        total_input_tokens=None,
        elapsed_ns=0,
        error=str(error),
    )


def main(argv: Sequence[str] | None = None) -> int:
    format_name = ""
    fixture_id = ""
    try:
        args = _parser().parse_args(argv)
        format_name = args.format
        fixture_id = args.fixture_id
        sample = run_sample(
            format_name=format_name,
            path=_path_or_none(args.path),
            source=_parse_source(args.source_json),
            options=_parse_options(args.options_json),
            fixture_id=fixture_id,
            seed=args.seed,
            model=args.model,
            tokenizer_name=args.tokenizer,
            apply_chat_template=args.apply_chat_template,
            exact_isl=args.exact_isl,
        )
    except Exception as error:
        sample = _error_sample(format_name, fixture_id, error)

    print(json.dumps(asdict(sample), separators=(",", ":")))
    return 1 if sample.error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
