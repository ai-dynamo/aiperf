# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-normalization and dataset-parse helpers for BenchmarkConfig.

These helpers keep the large Pydantic model validators short. They accept
and return plain dicts (pre-validation shape) and raise `ValueError` for
mutually-exclusive or malformed inputs.
"""

from __future__ import annotations

from typing import Any

_WARMUP_VS_PHASES_MSG = (
    "'warmup' cannot be used with 'phases'. "
    "Use 'warmup'/'profiling' for simple configs "
    "or 'phases' for advanced multi-phase configs."
)
_PROFILING_VS_PHASES_MSG = (
    "'profiling' cannot be used with 'phases'. "
    "Use 'warmup'/'profiling' for simple configs "
    "or 'phases' for advanced multi-phase configs."
)
DATASET_VS_DATASETS_MSG = (
    "'dataset' cannot be used with 'datasets'. "
    "Use 'dataset' for a single dataset "
    "or 'datasets' for multiple named datasets."
)
_MODEL_VS_MODELS_MSG = (
    "'model' cannot be used with 'models'. "
    "Use 'model' for a single model "
    "or 'models' for multiple named models."
)
_WARMUP_NEEDS_PROFILING_MSG = (
    "'warmup' requires 'profiling'. "
    "A warmup-only config without a profiling phase would produce no results."
)


def _check_mutual_exclusivity(data: dict[str, Any]) -> None:
    has_warmup = "warmup" in data
    has_profiling = "profiling" in data
    has_phases = "phases" in data

    if has_warmup and has_phases:
        raise ValueError(_WARMUP_VS_PHASES_MSG)
    if has_profiling and has_phases:
        raise ValueError(_PROFILING_VS_PHASES_MSG)
    if "dataset" in data and "datasets" in data:
        raise ValueError(DATASET_VS_DATASETS_MSG)
    if "model" in data and "models" in data:
        raise ValueError(_MODEL_VS_MODELS_MSG)
    if has_warmup and not has_profiling:
        raise ValueError(_WARMUP_NEEDS_PROFILING_MSG)


def _normalize_warmup_profiling_to_phases(data: dict[str, Any]) -> None:
    has_warmup = "warmup" in data
    has_profiling = "profiling" in data
    if not (has_warmup or has_profiling):
        return

    phases: list[dict[str, Any]] = []
    if has_warmup:
        warmup = data.pop("warmup")
        if isinstance(warmup, dict):
            warmup = {"name": "warmup", **warmup}
        phases.append(warmup)
    if has_profiling:
        prof = data.pop("profiling")
        if isinstance(prof, dict):
            prof = {"name": "profiling", **prof}
        phases.append(prof)
    data["phases"] = phases


def _normalize_models(data: dict[str, Any]) -> None:
    if "model" in data and "models" not in data:
        data["models"] = data.pop("model")

    if "models" not in data:
        return

    models = data["models"]
    if isinstance(models, str):
        data["models"] = {"items": [{"name": models}]}
    elif isinstance(models, list) and models and isinstance(models[0], str):
        data["models"] = {"items": [{"name": name} for name in models]}


def _normalize_dataset_and_phases(data: dict[str, Any]) -> None:
    if "dataset" in data and "datasets" not in data:
        ds = data.pop("dataset")
        if isinstance(ds, dict):
            ds = {"name": "default", **ds}
        data["datasets"] = [ds]

    if "phases" in data:
        phases = data["phases"]
        # Single flat-dict shorthand: phases: {type: concurrency, ...}
        if isinstance(phases, dict) and "type" in phases:
            data["phases"] = [{"name": "profiling", **phases}]


def normalize_benchmark_input(data: Any) -> Any:
    """Normalize BenchmarkConfig input before Pydantic validation.

    Handles:
        - model -> models (singular to plural)
        - dataset -> datasets (single-element list, ``name='default'`` injected)
        - phases: flat dict with 'type' -> single-element list, ``name='profiling'`` injected
        - models: str/list[str] -> ModelsAdvanced dict format
        - warmup/profiling -> phases
    """
    if not isinstance(data, dict):
        return data

    _check_mutual_exclusivity(data)
    _normalize_warmup_profiling_to_phases(data)
    _normalize_models(data)
    _normalize_dataset_and_phases(data)
    return data


def _hoist_synthetic_prompt_fields(config: dict[str, Any]) -> None:
    """Hoist top-level isl/osl into prompts.{isl,osl} for synthetic datasets."""
    ds_type = config.get("type")
    if ds_type not in ("synthetic", None):
        return
    if "isl" not in config and "osl" not in config:
        return

    prompts = config.setdefault("prompts", {})
    if not isinstance(prompts, dict):
        return
    if "isl" in config:
        prompts.setdefault("isl", config.pop("isl"))
    if "osl" in config:
        prompts.setdefault("osl", config.pop("osl"))


def _normalize_single_dataset_listed(
    idx: int, config: Any, dataset_types: tuple
) -> Any:
    # Accept already-constructed Pydantic models (for programmatic use)
    if isinstance(config, dataset_types):
        return config
    if not isinstance(config, dict):
        raise ValueError(
            f"datasets[{idx}] must be a dict (or a SyntheticDataset/FileDataset/"
            f"PublicDataset Pydantic model); got {type(config).__name__}. "
            f"Each entry needs at minimum a 'name' field; see docs/tutorials/yaml-config.md#datasets."
        )
    if "name" not in config:
        raise ValueError(
            f"datasets[{idx}] is missing required 'name' field. "
            f"Each dataset entry needs a name (e.g. 'main', 'eval')."
        )

    _hoist_synthetic_prompt_fields(config)

    if "type" not in config:
        config["type"] = "synthetic"
    return config


def parse_datasets_input(v: Any) -> list[Any]:
    """Parse dataset configurations into a list.

    Accepts already-constructed Pydantic models for programmatic use.
    """
    from aiperf.config.dataset import (
        FileDataset,
        PublicDataset,
        SyntheticDataset,
    )

    dataset_types = (SyntheticDataset, FileDataset, PublicDataset)

    if not isinstance(v, list):
        raise ValueError(
            f"datasets must be a list of named entries (was a dict in earlier versions); "
            f"got {type(v).__name__}. Use [{{'name': 'main', 'type': 'synthetic', ...}}, ...]. "
            f"See docs/tutorials/yaml-config.md#datasets."
        )

    return [
        _normalize_single_dataset_listed(idx, item, dataset_types)
        for idx, item in enumerate(v)
    ]
