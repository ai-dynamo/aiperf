# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned projection from Config v2 into the native single-run contract.

Config v2 remains the user-facing and orchestration schema.  This module is
the only place where a fully resolved :class:`BenchmarkRun` is lowered into
the narrower Rust execution ABI; raw Pydantic dumps are deliberately not a
process boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.config.dataset import FileDataset, PublicDataset, SyntheticDataset
from aiperf.config.phases import (
    ConcurrencyPhase,
    ConstantPhase,
    FixedSchedulePhase,
    GammaPhase,
    PoissonPhase,
    UserCentricPhase,
)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


RUNNER_PROTOCOL_VERSION = 1


class RustWireError(ValueError):
    """Raised when a resolved Config v2 run cannot enter the native ABI."""


def build_run_request(run: BenchmarkRun) -> dict[str, Any]:
    """Build the complete protocol-v1 request for one native benchmark.

    Every accepted field is written explicitly.  That makes additions to
    Config v2 fail closed until this projection and the Rust DTO are updated in
    the same change.
    """
    cfg = run.cfg
    dataset = cfg.get_default_dataset()

    models = [
        {
            "name": item.name,
            **({"weight": item.weight} if item.weight is not None else {}),
        }
        for item in cfg.models.items
    ]
    endpoint = cfg.endpoint
    endpoint_wire: dict[str, Any] = {
        "urls": list(endpoint.urls),
        "type": str(endpoint.type),
        "streaming": endpoint.streaming,
        "use_legacy_max_tokens": endpoint.use_legacy_max_tokens,
        "use_server_token_count": endpoint.use_server_token_count,
        "timeout_seconds": endpoint.timeout,
        "extra": dict(endpoint.extra),
        "headers": dict(endpoint.headers),
        "http2": False,
    }
    _set_optional(endpoint_wire, "path", endpoint.path)
    _set_optional(endpoint_wire, "api_key", endpoint.api_key)
    _set_optional(endpoint_wire, "session_header", endpoint.session_header)
    if endpoint.template is not None:
        endpoint_wire["template"] = endpoint.template.body
        endpoint_wire["response_field"] = endpoint.template.response_field

    variation = run.variation
    run_wire: dict[str, Any] = {
        "benchmark_id": run.benchmark_id,
        "label": run.label,
        "trial": run.trial,
        "artifact_dir": str(run.artifact_dir),
        "models": {"strategy": str(cfg.models.strategy), "items": models},
        "endpoint": endpoint_wire,
        "dataset": _dataset(run, dataset),
        "tokenizer": {
            "name": _tokenizer_source(run),
            **(
                {"apply_chat_template": True}
                if cfg.tokenizer is not None and cfg.tokenizer.apply_chat_template
                else {}
            ),
        },
        "phases": [_phase(phase) for phase in cfg.phases],
        "metrics": {
            "slos": dict(cfg.slos or {}),
            **(
                {"slice_duration_seconds": cfg.artifacts.slice_duration}
                if cfg.artifacts.slice_duration is not None
                else {}
            ),
        },
        "artifacts": {
            **(
                {
                    "records_path": _artifact_relative_path(
                        run.artifact_dir,
                        cfg.artifacts.profile_export_jsonl_file,
                    )
                }
                if cfg.artifacts.records is not False or cfg.artifacts.raw
                else {}
            ),
            "trace": cfg.artifacts.trace,
        },
    }
    _set_optional(run_wire, "sweep_id", run.sweep_id)
    _set_optional(run_wire, "random_seed", run.random_seed)
    if variation is not None:
        run_wire["variation"] = {
            "index": variation.index,
            "label": variation.label,
            "values": dict(variation.values),
        }
    return {"protocol_version": RUNNER_PROTOCOL_VERSION, "run": run_wire}


def _dataset(run: BenchmarkRun, dataset: Any) -> dict[str, Any]:
    if isinstance(dataset, SyntheticDataset):
        return _synthetic_dataset(dataset)
    if isinstance(dataset, FileDataset):
        return _file_dataset(run, dataset)
    if isinstance(dataset, PublicDataset):
        return _public_dataset(run, dataset)
    raise RustWireError(
        f"native runner protocol v1 does not accept dataset type {dataset.type!s}"
    )


def _synthetic_dataset(dataset: SyntheticDataset) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "synthetic",
        "entries": dataset.entries,
        "sampling": str(dataset.sampling),
        "turns": _distribution(dataset.turns or 1),
        "turn_delay_ms": _distribution(dataset.turn_delay or 0),
        "turn_delay_ratio": dataset.turn_delay_ratio,
    }
    _set_optional(result, "random_seed", dataset.random_seed)
    if dataset.prompts is not None:
        prompts: dict[str, Any] = {"batch_size": dataset.prompts.batch_size}
        if dataset.prompts.isl is not None:
            prompts["isl"] = _distribution(dataset.prompts.isl)
        if dataset.prompts.osl is not None:
            prompts["osl"] = _distribution(dataset.prompts.osl)
        _set_optional(prompts, "block_size", dataset.prompts.block_size)
        if dataset.prompts.sequence_distribution is not None:
            prompts["sequence_distribution"] = [
                {
                    "isl": _distribution(entry.isl),
                    "osl": _distribution(entry.osl),
                    "probability": entry.probability,
                }
                for entry in dataset.prompts.sequence_distribution
            ]
        result["prompts"] = prompts
    if dataset.prefix_prompts is not None:
        result["prefix_prompts"] = dataset.prefix_prompts.model_dump(
            mode="json", exclude_none=True
        )
    if dataset.images is not None:
        source = dataset.images.source
        source_value = (
            str(source.expanduser().resolve())
            if isinstance(source, Path)
            else str(source)
        )
        result["images"] = {
            "batch_size": dataset.images.batch_size,
            "width": _distribution(dataset.images.width),
            "height": _distribution(dataset.images.height),
            "format": str(dataset.images.format),
            "source": source_value,
            "source_sampling": str(dataset.images.source_sampling),
        }
    if dataset.audio is not None:
        result["audio"] = {
            "batch_size": dataset.audio.batch_size,
            "length": _distribution(dataset.audio.length),
            "format": str(dataset.audio.format),
            "sample_rates": list(dataset.audio.sample_rates),
            "depths": list(dataset.audio.depths),
            "channels": dataset.audio.channels,
        }
    if dataset.video is not None:
        video: dict[str, Any] = {
            "batch_size": dataset.video.batch_size,
            "duration": dataset.video.duration,
            "fps": dataset.video.fps,
            "format": str(dataset.video.format),
            "codec": dataset.video.codec,
            "synth_type": str(dataset.video.synth_type),
            "audio": {
                "sample_rate": dataset.video.audio.sample_rate,
                "channels": dataset.video.audio.channels,
                "depth": dataset.video.audio.depth,
            },
        }
        _set_optional(video, "width", dataset.video.width)
        _set_optional(video, "height", dataset.video.height)
        _set_optional(video["audio"], "codec", dataset.video.audio.codec)
        result["video"] = video
    if dataset.rankings is not None:
        result["rankings"] = {
            "passages": _distribution(dataset.rankings.passages),
            "passage_tokens": _distribution(dataset.rankings.passage_tokens),
            "query_tokens": _distribution(dataset.rankings.query_tokens),
        }
    return result


def _file_dataset(run: BenchmarkRun, dataset: FileDataset) -> dict[str, Any]:
    resolved_types = run.resolved.dataset_types or {}
    resolved_sampling = run.resolved.dataset_sampling_strategies or {}
    format_name = str(resolved_types.get(dataset.name, dataset.format))
    native_format, format_options = _native_file_format(format_name)
    if native_format == "mooncake_trace":
        format_options.setdefault("block_size", 512)
    elif native_format == "bailian_trace":
        format_options.setdefault("block_size", 16)
    if dataset.inter_turn_delay_cap_seconds is not None:
        format_options["inter_turn_delay_cap_seconds"] = (
            dataset.inter_turn_delay_cap_seconds
        )
    result: dict[str, Any] = {
        "type": "file",
        "format": native_format,
        "sampling": str(resolved_sampling.get(dataset.name, dataset.sampling)),
        "options": format_options,
    }
    _set_optional(result, "entries", dataset.entries)
    _set_optional(result, "random_seed", dataset.random_seed)
    if dataset.osl is not None:
        result["osl"] = _distribution(dataset.osl)
    if dataset.synthesis is not None:
        result["synthesis"] = dataset.synthesis.model_dump(
            mode="json", exclude_none=True
        )
    if dataset.path is not None:
        resolved_paths = run.resolved.dataset_file_paths or {}
        path = Path(resolved_paths.get(dataset.name, dataset.path)).resolve()
        result["path"] = str(path)
    else:
        result["records"] = dataset.records
    return result


_PUBLIC_NATIVE_FORMATS = {
    "aiperf.dataset.loader.exgentic:ExgenticDatasetLoader": "exgentic",
    "aiperf.dataset.loader.exgentic_v2:ExgenticV2DatasetLoader": "exgentic_v2",
    "aiperf.dataset.loader.sharegpt:ShareGPTLoader": "sharegpt",
    "aiperf.dataset.loader.hf_instruction_response:HFInstructionResponseDatasetLoader": (
        "hf_instruction_response"
    ),
    "aiperf.dataset.loader.hf_conversation:HFConversationDatasetLoader": (
        "hf_conversation"
    ),
    "aiperf.dataset.loader.mt_bench:MTBenchDatasetLoader": "mt_bench",
    "aiperf.dataset.loader.mmvu:MMVUDatasetLoader": "mmvu",
    "aiperf.dataset.loader.spec_bench:SpecBenchLoader": "spec_bench",
    "aiperf.dataset.loader.hf_asr:HFASRDatasetLoader": "hf_asr",
}


def _public_dataset(run: BenchmarkRun, dataset: PublicDataset) -> dict[str, Any]:
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    loader_class = plugins.get_class(PluginType.PUBLIC_DATASET_LOADER, dataset.dataset)
    class_key = f"{loader_class.__module__}:{loader_class.__name__}"
    try:
        native_format = _PUBLIC_NATIVE_FORMATS[class_key]
    except KeyError as error:
        raise RustWireError(
            f"public dataset {dataset.dataset!s} uses loader {class_key!r}, "
            "which has no native loader registration"
        ) from error
    metadata = plugins.get_public_dataset_loader_metadata(dataset.dataset)
    options: dict[str, Any] = {}
    for name in (
        "prompt_column",
        "image_column",
        "video_column",
        "audio_column",
        "prompt_template",
        "conversation_column",
    ):
        _set_optional(options, name, getattr(metadata, name))
    if metadata.conversation_column is not None:
        options["message_content_key"] = metadata.message_content_key
    if metadata.multi_turn:
        options["multi_turn"] = True
    if dataset.filters:
        if native_format not in {"exgentic", "exgentic_v2"}:
            raise RustWireError(
                f"public dataset {dataset.dataset!s} does not accept dataset filters"
            )
        options.update(dataset.filters)
    if native_format in {"exgentic", "exgentic_v2"}:
        options["fixed_schedule"] = any(
            isinstance(phase, FixedSchedulePhase) for phase in run.cfg.phases
        )

    max_conversations = _public_max_conversations(
        run,
        dataset,
        streaming=metadata.streaming,
        entries_first=native_format in {"exgentic", "exgentic_v2"},
    )
    if native_format in {"exgentic", "exgentic_v2"} and max_conversations is None:
        raise RustWireError(
            "Exgentic requires a finite entries or profiling request count"
        )
    if max_conversations is not None:
        options["max_conversations"] = max_conversations

    if metadata.hf_dataset_name is not None:
        source: dict[str, Any] = {
            "type": "hugging_face",
            "dataset": metadata.hf_dataset_name,
            "subset": dataset.hf_subset or metadata.hf_subset or "default",
            "split": metadata.hf_split,
        }
        _set_optional(source, "revision", getattr(loader_class, "hf_revision", None))
    else:
        url = getattr(loader_class, "url", None)
        if not isinstance(url, str) or not url:
            raise RustWireError(
                f"public dataset {dataset.dataset!s} has neither Hugging Face "
                "coordinates nor a loader URL"
            )
        source = {"type": "url", "url": url}

    result: dict[str, Any] = {
        "type": "public",
        "name": str(dataset.dataset),
        "format": native_format,
        "source": source,
        "sampling": str(dataset.sampling),
        "options": options,
    }
    _set_optional(result, "entries", dataset.entries)
    _set_optional(result, "random_seed", dataset.random_seed)
    return result


def _public_max_conversations(
    run: BenchmarkRun,
    dataset: PublicDataset,
    *,
    streaming: bool,
    entries_first: bool,
) -> int | None:
    request_counts = [
        phase.requests
        for phase in run.cfg.get_profiling_phases()
        if phase.requests is not None
    ]
    request_cap = max(request_counts) if request_counts else None
    if entries_first and dataset.entries is not None:
        return dataset.entries
    if streaming and request_cap is not None:
        return request_cap
    return dataset.entries


def _native_file_format(format_name: str) -> tuple[str, dict[str, Any]]:
    if format_name == "burst_gpt_trace":
        return "burst_gpt", {}
    if not format_name.startswith("speed_bench_"):
        return format_name, {}
    suffix = format_name.removeprefix("speed_bench_")
    category = None
    for candidate in (
        "low_entropy",
        "mixed",
        "high_entropy",
        "coding",
        "humanities",
        "math",
        "multilingual",
        "qa",
        "rag",
        "reasoning",
        "roleplay",
        "stem",
        "summarization",
        "writing",
    ):
        if suffix == candidate or suffix.endswith(f"_{candidate}"):
            category = candidate
            break
    return "speed_bench", ({"category": category} if category else {})


def _tokenizer_source(run: BenchmarkRun) -> str:
    cfg = run.cfg.tokenizer
    primary_model = run.cfg.models.items[0].name
    resolved = run.resolved.tokenizer_names or {}
    name = resolved.get(primary_model) or (cfg.name if cfg is not None else None)
    if name is None:
        from aiperf.common.tokenizer_fake_names import is_fake_model_name

        name = "builtin" if is_fake_model_name(primary_model) else primary_model
    normalized = name.lower().replace("-", "_")
    if normalized in {
        "builtin",
        "o200k_base",
        "o200k_harmony",
        "cl100k_base",
        "p50k_base",
        "p50k_edit",
        "r50k_base",
    }:
        return normalized
    path = Path(name).expanduser()
    if path.exists():
        return str(path.resolve())
    try:
        from huggingface_hub import try_to_load_from_cache

        tokenizer_file = try_to_load_from_cache(
            name,
            "tokenizer.json",
            revision=cfg.revision if cfg is not None else "main",
        )
    except (ImportError, OSError, ValueError) as error:
        raise RustWireError(
            f"cannot resolve native tokenizer.json for {name!r}: {error}"
        ) from error
    if not isinstance(tokenizer_file, str):
        raise RustWireError(
            f"Python resolved tokenizer {name!r}, but its tokenizer.json is not cached"
        )
    return str(Path(tokenizer_file).resolve().parent)


def _phase(phase: Any) -> dict[str, Any]:
    common: dict[str, Any] = {
        "name": phase.name,
        "exclude_from_results": phase.exclude_from_results,
        "seamless": phase.seamless,
    }
    for name in (
        "requests",
        "sessions",
        "duration",
        "prefill_concurrency",
        "grace_period",
    ):
        _set_optional(common, name, getattr(phase, name))
    _set_optional(common, "concurrency_ramp", _ramp(phase.concurrency_ramp))
    _set_optional(common, "prefill_ramp", _ramp(phase.prefill_ramp))
    _set_optional(common, "rate_ramp", _ramp(getattr(phase, "rate_ramp", None)))
    if phase.cancellation is not None:
        common["cancellation"] = {
            "rate": phase.cancellation.rate,
            "delay": phase.cancellation.delay,
        }
    adaptive_scale = _adaptive_scale(phase)
    if adaptive_scale is not None:
        common["adaptive_scale"] = adaptive_scale

    if isinstance(phase, ConcurrencyPhase):
        return {"type": "concurrency", **common, "concurrency": phase.concurrency}
    if isinstance(phase, PoissonPhase):
        return _rate_phase("poisson", phase, common)
    if isinstance(phase, GammaPhase):
        result = _rate_phase("gamma", phase, common)
        _set_optional(result, "smoothness", phase.smoothness)
        return result
    if isinstance(phase, ConstantPhase):
        return _rate_phase("constant", phase, common)
    if isinstance(phase, UserCentricPhase):
        result = {
            "type": "user_centric",
            **common,
            "rate": phase.rate,
            "users": phase.users,
        }
        _set_optional(result, "concurrency", phase.concurrency)
        return result
    if isinstance(phase, FixedSchedulePhase):
        result = {
            "type": "fixed_schedule",
            **common,
            "auto_offset": phase.auto_offset,
        }
        _set_optional(result, "start_offset", phase.start_offset)
        _set_optional(result, "end_offset", phase.end_offset)
        return result
    raise RustWireError(
        f"native runner protocol v1 does not accept phase type {phase.type!s}"
    )


def _rate_phase(kind: str, phase: Any, common: dict[str, Any]) -> dict[str, Any]:
    result = {"type": kind, **common, "rate": phase.rate}
    _set_optional(result, "concurrency", phase.concurrency)
    return result


def _adaptive_scale(phase: Any) -> dict[str, Any] | None:
    enabled = bool(getattr(phase, "adaptive_scale", False))
    sla_filters = list(getattr(phase, "sla", ()) or ())
    if not enabled:
        if sla_filters:
            raise RustWireError(
                f"phase {phase.name!r} defines adaptive SLA filters without "
                "enabling adaptive_scale"
            )
        return None
    if phase.name != "profiling":
        raise RustWireError("adaptive_scale is supported only on profiling phases")

    variable = str(phase.adaptive_control_variable)
    maximum = phase.adaptive_control_max
    if maximum is None:
        maximum = {
            "concurrency": phase.concurrency,
            "prefill_concurrency": phase.prefill_concurrency,
            "request_rate": getattr(phase, "rate", None),
            "users": getattr(phase, "users", None),
        }.get(variable)
    if maximum is None:
        raise RustWireError(
            f"adaptive_scale control.max could not be resolved for {variable!r}"
        )

    return {
        "control_variable": variable,
        "minimum": phase.adaptive_control_min,
        "maximum": maximum,
        "assessment_period_seconds": phase.adaptive_assessment_period or 30.0,
        "sustain_duration_seconds": phase.adaptive_sustain_duration,
        "min_completed_requests": phase.adaptive_min_completed_requests,
        "strategy_type": phase.adaptive_scale_strategy_type,
        "step_policy": phase.adaptive_scale_step_policy,
        "base_step": phase.adaptive_scale_base_step,
        "max_step_multiplier": phase.adaptive_scale_max_step_multiplier,
        "step_percent": phase.adaptive_scale_step_percent,
        "sla_filters": [
            {
                "metric_tag": sla.metric_tag,
                "stat": sla.stat,
                "op": sla.op,
                "threshold": sla.threshold,
            }
            for sla in sla_filters
        ],
    }


def _ramp(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"duration": value.duration, "strategy": str(value.strategy)}


def _distribution(value: Any) -> dict[str, Any]:
    if isinstance(value, int | float):
        return {"value": float(value)}
    dumped = value.model_dump(mode="json", exclude_none=True)
    if "peaks" in dumped:
        dumped["peaks"] = [
            {
                "distribution": _distribution(peak.distribution),
                "weight": peak.weight,
            }
            for peak in value.peaks
        ]
    return dumped


def _set_optional(target: dict[str, Any], name: str, value: Any) -> None:
    if value is not None:
        target[name] = value


def _artifact_relative_path(root: Path, output: Path) -> str:
    root_path = root.resolve()
    output_path = output.resolve()
    try:
        return str(output_path.relative_to(root_path))
    except ValueError as error:
        raise RustWireError(
            f"native artifact path {output_path} is outside run directory {root_path}"
        ) from error
