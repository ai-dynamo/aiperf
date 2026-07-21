# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin Python adapter for dataset load, composition, and tokenization timing."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aiperf.common import random_generator as rng
from aiperf.common.enums import MediaType
from aiperf.common.models import Conversation
from aiperf.common.tokenizer import Tokenizer
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.bailian_trace import BailianTraceDatasetLoader
from aiperf.dataset.loader.burst_gpt import BurstGPTTraceDatasetLoader
from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader
from aiperf.dataset.loader.random_pool import RandomPoolDatasetLoader
from aiperf.dataset.loader.raw_payload import RawPayloadDatasetLoader
from aiperf.dataset.loader.sagemaker_data_capture import SageMakerDataCaptureLoader
from aiperf.dataset.loader.single_turn import SingleTurnDatasetLoader
from aiperf.endpoints.payload_extraction import extract_inputs

LOADERS = {
    "single_turn": SingleTurnDatasetLoader,
    "multi_turn": MultiTurnDatasetLoader,
    "raw_payload": RawPayloadDatasetLoader,
    "inputs_json": InputsJsonPayloadLoader,
    "random_pool": RandomPoolDatasetLoader,
    "mooncake_trace": MooncakeTraceDatasetLoader,
    "bailian_trace": BailianTraceDatasetLoader,
    "burst_gpt_trace": BurstGPTTraceDatasetLoader,
    "sagemaker_data_capture": SageMakerDataCaptureLoader,
}

NON_EMPTY_OPTIONS_REASON = (
    "non-empty options are unsupported until cross-stack option mapping is verified"
)

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


def _benchmark_run(*, seed: int, model: str) -> BenchmarkRun:
    cfg = BenchmarkConfig.model_validate(
        {
            "models": [model],
            "endpoint": {
                "urls": ["http://localhost:8000/v1/chat/completions"],
                "wait_for_model_timeout": 0,
            },
            "datasets": [
                {
                    "name": "adapter",
                    "type": "synthetic",
                    "entries": 1,
                    "prompts": {"isl": 1, "osl": 1},
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 1,
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
    return sum(len(rows) for rows in loaded.values())


def _count_input_tokens(
    conversations: list[Conversation],
    tokenizer: Tokenizer,
) -> int:
    total = 0
    for conversation in conversations:
        for turn in conversation.turns:
            for text in turn.texts:
                for content in text.contents:
                    total += len(tokenizer.encode(content))
            if turn.raw_payload is not None:
                extracted = extract_inputs(turn.raw_payload, _CHAT_PART_TYPES)
                total += extracted.pretokenised_token_count
                total += sum(len(tokenizer.encode(text)) for text in extracted.texts)
            if turn.raw_messages is not None:
                payload: dict[str, object] = {"messages": turn.raw_messages}
                if turn.raw_tools is not None:
                    payload["tools"] = turn.raw_tools
                extracted = extract_inputs(payload, _CHAT_PART_TYPES)
                total += extracted.pretokenised_token_count
                total += sum(len(tokenizer.encode(text)) for text in extracted.texts)
    return total


def _create_loader(
    format_name: str,
    path: Path,
    run: BenchmarkRun,
    tokenizer: Tokenizer,
) -> Any:
    """Construct a loader with the same trace dependencies as the custom composer."""
    loader_class = LOADERS[format_name]
    kwargs: dict[str, object] = {}
    if format_name in {
        "mooncake_trace",
        "bailian_trace",
        "burst_gpt_trace",
        "sagemaker_data_capture",
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
    return loader_class(filename=path, run=run, **kwargs)


def _total_input_tokens(
    format_name: str,
    loaded: dict[str, list[Any]],
    conversations: list[Conversation],
    tokenizer: Tokenizer,
) -> int | None:
    if format_name in {"raw_payload", "inputs_json"}:
        return None
    if format_name in {"bailian_trace", "burst_gpt_trace"}:
        return sum(
            trace.input_length for traces in loaded.values() for trace in traces
        )
    return _count_input_tokens(conversations, tokenizer)


def run_sample(
    *,
    format_name: str,
    path: Path,
    options: dict[str, object],
    fixture_id: str,
    seed: int,
    model: str,
) -> Sample:
    """Measure one load, compose, and tokenize operation."""
    loader_class = LOADERS.get(format_name)
    if loader_class is None:
        raise ValueError(f"unsupported dataset format: {format_name}")
    if options:
        raise ValueError(NON_EMPTY_OPTIONS_REASON)

    # The production bootstrap initializes this manager before composer/loader
    # construction. Each adapter invocation is one isolated measurement, so
    # resetting here gives repeatable construction from the requested seed.
    rng.reset()
    rng.init(seed)
    run = _benchmark_run(seed=seed, model=model)
    tokenizer = Tokenizer.from_pretrained("builtin")
    # Match the Rust adapter: warm encoding tables outside the timed region.
    tokenizer.encode("warm")
    loader = _create_loader(format_name, path, run, tokenizer)

    started_ns = time.perf_counter_ns()
    loaded = loader.load_dataset()
    conversations = loader.convert_to_conversations(loaded)
    total_input_tokens = _total_input_tokens(
        format_name, loaded, conversations, tokenizer
    )
    elapsed_ns = time.perf_counter_ns() - started_ns

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
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--options-json", required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", required=True)
    return parser


def _parse_options(raw: str) -> dict[str, object]:
    options = json.loads(raw)
    if not isinstance(options, dict):
        raise ValueError("--options-json must decode to a JSON object")
    return options


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
            path=args.path,
            options=_parse_options(args.options_json),
            fixture_id=fixture_id,
            seed=args.seed,
            model=args.model,
        )
    except Exception as error:
        sample = _error_sample(format_name, fixture_id, error)

    print(json.dumps(asdict(sample), separators=(",", ":")))
    return 1 if sample.error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
