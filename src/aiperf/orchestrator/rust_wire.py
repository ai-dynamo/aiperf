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

from aiperf.config.dataset import SyntheticDataset
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
    if not isinstance(dataset, SyntheticDataset):
        raise RustWireError(
            f"native runner protocol v1 does not accept dataset type {dataset.type!s}"
        )
    if dataset.prompts is None or dataset.prompts.isl is None:
        raise RustWireError("synthetic native runs require datasets[].prompts.isl")
    if dataset.prompts.osl is None:
        raise RustWireError("synthetic native runs require datasets[].prompts.osl")

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
        "dataset": {
            "type": "synthetic",
            "entries": dataset.entries,
            "prompts": {
                "isl": _distribution(dataset.prompts.isl),
                "osl": _distribution(dataset.prompts.osl),
                "batch_size": dataset.prompts.batch_size,
            },
            "turns": _distribution(dataset.turns or 1),
            "turn_delay_ms": _distribution(dataset.turn_delay or 0),
            "turn_delay_ratio": dataset.turn_delay_ratio,
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
