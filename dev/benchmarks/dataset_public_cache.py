# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Untimed public-source prefetch for the dataset-load comparison harness."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import orjson

from dev.benchmarks.dataset_format_catalog import (
    DEFAULT_BENCHMARK_ROWS,
    FormatCase,
    SourceEnvelope,
    profile_for,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DATASETS_PATH = REPOSITORY_ROOT / "rust/cli/resources/public_datasets.json"


def load_public_dataset_catalog() -> dict[str, Any]:
    return json.loads(PUBLIC_DATASETS_PATH.read_bytes())


def _public_entry(pin_key: str) -> dict[str, Any]:
    catalog = load_public_dataset_catalog()
    try:
        entry = catalog[pin_key]
    except KeyError as error:
        raise ValueError(f"unknown public dataset pin {pin_key!r}") from error
    if not isinstance(entry, dict):
        raise ValueError(f"public dataset pin {pin_key!r} must be an object")
    return entry


def _identity_for_entry(pin_key: str, entry: dict[str, Any]) -> dict[str, Any]:
    source = entry.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"public pin {pin_key!r} missing source object")
    identity: dict[str, Any] = {"pin_key": pin_key, "format": entry.get("format")}
    source_type = source.get("type")
    identity["source_type"] = source_type
    if source_type == "hugging_face":
        identity["dataset"] = source.get("dataset")
        identity["subset"] = source.get("subset")
        identity["split"] = source.get("split")
        identity["revision"] = source.get("revision")
    elif source_type == "url":
        url = str(source.get("url", ""))
        identity["url"] = url
        identity["url_sha256"] = hashlib.sha256(url.encode()).hexdigest()
    else:
        raise ValueError(f"unsupported public source type {source_type!r}")
    return identity


def _merge_options(
    entry: dict[str, Any], case_options: dict[str, object]
) -> dict[str, Any]:
    metadata = entry.get("options")
    merged: dict[str, Any] = dict(metadata) if isinstance(metadata, dict) else {}
    merged.update(case_options)
    return merged


async def _prefetch_url_loader(
    loader_class: type,
    *,
    run: Any,
    tokenizer: Any,
    loader_kwargs: dict[str, Any],
) -> None:
    loader = loader_class(run=run, tokenizer=tokenizer, **loader_kwargs)
    await loader.load_dataset()


async def _prefetch_hf_loader(
    loader_class: type,
    *,
    run: Any,
    entry: dict[str, Any],
    case_options: dict[str, object],
    profile: Any,
) -> None:
    source = entry["source"]
    loader_kwargs: dict[str, Any] = {
        "hf_dataset_name": source["dataset"],
        "hf_split": source.get("split", "train"),
        "hf_subset": source.get("subset"),
        "streaming": effective_streaming(profile, entry),
    }
    revision = source.get("revision")
    if revision:
        loader_kwargs["hf_revision"] = revision
    loader_kwargs.update(_merge_options(entry, case_options))
    loader = loader_class(run=run, **loader_kwargs)
    await loader.load_dataset()


def _benchmark_run_for_prefetch(*, seed: int, entries: int) -> Any:
    from aiperf.config import BenchmarkConfig, BenchmarkRun

    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {
                "urls": ["http://localhost:8000/v1/chat/completions"],
                "wait_for_model_timeout": 0,
            },
            "datasets": [
                {
                    "name": "prefetch",
                    "type": "synthetic",
                    "entries": entries,
                    "prompts": {"isl": 1, "osl": 1},
                }
            ],
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
        benchmark_id="dataset-load-prefetch",
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=seed,
        cli_command=None,
    )


def effective_streaming(profile: Any, entry: dict[str, Any]) -> bool:
    """Resolve the streaming flag used by the harness for a public pin.

    A ``snapshot_cache`` pin is a streaming source that HF cannot resolve from a
    partial cache offline, so the harness loads it non-streaming in both the
    prefetch and timed phases; that materializes the ``datasets`` arrow cache the
    offline timed run reads, matching every other non-streaming HF family.
    """
    if getattr(profile, "snapshot_cache", False):
        return False
    return bool(entry.get("streaming", False))


async def _prefetch_python_case(case: FormatCase, *, seed: int, row_limit: int) -> None:
    profile = profile_for(case.format)
    if profile is None or profile.source_kind != "public_cached":
        return
    pin_key = case.source.public.get("pin_key") if case.source.public else None
    if not isinstance(pin_key, str):
        raise ValueError(f"{case.format} missing public pin_key")
    entry = _public_entry(pin_key)
    from aiperf.common import random_generator as rng

    rng.reset()
    rng.init(seed)
    run = _benchmark_run_for_prefetch(seed=seed, entries=row_limit)
    from aiperf.common.tokenizer import Tokenizer

    tokenizer = Tokenizer.from_pretrained("builtin")
    options = _merge_options(entry, case.options)

    format_name = str(entry.get("format", case.format))
    if format_name == "sharegpt":
        from aiperf.dataset.loader.sharegpt import ShareGPTLoader

        await _prefetch_url_loader(
            ShareGPTLoader, run=run, tokenizer=tokenizer, loader_kwargs={}
        )
        return
    if format_name == "spec_bench":
        from aiperf.dataset.loader.spec_bench import SpecBenchLoader

        await _prefetch_url_loader(
            SpecBenchLoader,
            run=run,
            tokenizer=tokenizer,
            loader_kwargs={"multi_turn": False},
        )
        return
    if format_name == "hf_instruction_response":
        from aiperf.dataset.loader.hf_instruction_response import (
            HFInstructionResponseDatasetLoader,
        )

        await _prefetch_hf_loader(
            HFInstructionResponseDatasetLoader,
            run=run,
            entry=entry,
            case_options=options,
            profile=profile,
        )
        return
    if format_name == "hf_conversation":
        from aiperf.dataset.loader.hf_conversation import HFConversationDatasetLoader

        await _prefetch_hf_loader(
            HFConversationDatasetLoader,
            run=run,
            entry=entry,
            case_options=options,
            profile=profile,
        )
        return
    if format_name == "hf_asr":
        from aiperf.dataset.loader.hf_asr import HFASRDatasetLoader

        await _prefetch_hf_loader(
            HFASRDatasetLoader,
            run=run,
            entry=entry,
            case_options=options,
            profile=profile,
        )
        return
    if format_name == "mt_bench":
        from aiperf.dataset.loader.mt_bench import MTBenchDatasetLoader

        await _prefetch_hf_loader(
            MTBenchDatasetLoader,
            run=run,
            entry=entry,
            case_options=options,
            profile=profile,
        )
        return
    if format_name == "mmvu":
        from aiperf.dataset.loader.mmvu import MMVUDatasetLoader

        await _prefetch_hf_loader(
            MMVUDatasetLoader,
            run=run,
            entry=entry,
            case_options=options,
            profile=profile,
        )
        return
    if format_name == "exgentic":
        from aiperf.dataset.loader.exgentic import ExgenticDatasetLoader

        await _prefetch_hf_loader(
            ExgenticDatasetLoader,
            run=run,
            entry=entry,
            case_options={**options, "max_conversations": row_limit},
            profile=profile,
        )
        return
    if format_name == "exgentic_v2":
        from aiperf.dataset.loader.exgentic_v2 import ExgenticV2DatasetLoader

        await _prefetch_hf_loader(
            ExgenticV2DatasetLoader,
            run=run,
            entry=entry,
            case_options={**options, "max_conversations": row_limit},
            profile=profile,
        )
        return
    raise ValueError(f"unsupported public format {format_name!r}")


def prefetch_public_case(
    case: FormatCase,
    *,
    seed: int = 42,
    row_limit: int = DEFAULT_BENCHMARK_ROWS,
) -> FormatCase:
    """Download/cache one public case and return it with recorded source identity."""
    if case.source.kind != "public_cached":
        return case
    asyncio.run(_prefetch_python_case(case, seed=seed, row_limit=row_limit))
    pin_key = case.source.public["pin_key"]  # type: ignore[index]
    entry = _public_entry(str(pin_key))
    identity = _identity_for_entry(str(pin_key), entry)
    identity["row_limit"] = row_limit
    public = dict(case.source.public or {})
    public["identity"] = identity
    return FormatCase(
        format=case.format,
        fixture_id=case.fixture_id,
        options=dict(case.options),
        source=SourceEnvelope(kind="public_cached", public=public),
    )


def prefetch_public_cases(
    cases: list[FormatCase],
    *,
    seed: int = 42,
    row_limit: int = DEFAULT_BENCHMARK_ROWS,
) -> tuple[list[FormatCase], list[dict[str, str]]]:
    """Prefetch every public case in ``cases`` outside the timed region.

    A public source that cannot be fetched (for example a gated HuggingFace
    dataset that requires interactive authentication) is recorded as a skip and
    excluded from the timed comparison rather than aborting the whole run.
    """
    prepared: list[FormatCase] = []
    skips: list[dict[str, str]] = []
    for case in cases:
        if case.source.kind != "public_cached":
            prepared.append(case)
            continue
        try:
            prepared.append(prefetch_public_case(case, seed=seed, row_limit=row_limit))
        except Exception as error:  # noqa: BLE001 - prefetch is best-effort
            skips.append(
                {
                    "format": case.format,
                    "reason": f"public prefetch failed: {error}",
                }
            )
    return prepared, skips


OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def public_source_for_rust(case: FormatCase) -> dict[str, Any]:
    """Build the Rust-facing public source payload from a prepared case."""
    if case.source.kind != "public_cached":
        raise ValueError("public_source_for_rust requires a public_cached case")
    public = case.source.public or {}
    pin_key = public.get("pin_key")
    if not isinstance(pin_key, str):
        raise ValueError("missing public pin_key")
    entry = _public_entry(pin_key)
    source = dict(entry["source"])
    profile = profile_for(case.format)
    streaming = (
        effective_streaming(profile, entry)
        if profile is not None
        else bool(entry.get("streaming", False))
    )
    payload: dict[str, Any] = {
        "pin_key": pin_key,
        "format": entry.get("format", case.format),
        "source": source,
        "options": _merge_options(entry, case.options),
        "streaming": streaming,
        "identity": public.get("identity"),
    }
    return payload


def source_json_for_adapters(case: FormatCase) -> str:
    """Serialize the normalized source envelope for adapter CLIs."""
    payload = case.source.to_dict()
    if case.source.kind == "public_cached":
        payload["rust_public"] = public_source_for_rust(case)
    return orjson.dumps(payload).decode()
