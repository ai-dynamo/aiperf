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
_DATASET_VS_DATASETS_MSG = (
    "'dataset' cannot be used with 'datasets'. "
    "Use 'dataset' for a single dataset "
    "or 'datasets' for multiple named datasets."
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
        raise ValueError(_DATASET_VS_DATASETS_MSG)
    if has_warmup and not has_profiling:
        raise ValueError(_WARMUP_NEEDS_PROFILING_MSG)


def _normalize_warmup_profiling_to_phases(data: dict[str, Any]) -> None:
    has_warmup = "warmup" in data
    has_profiling = "profiling" in data
    if not (has_warmup or has_profiling):
        return

    phases: dict[str, Any] = {}
    if has_warmup:
        warmup = data.pop("warmup")
        if isinstance(warmup, dict):
            warmup.setdefault("exclude_from_results", True)
        phases["warmup"] = warmup
    if has_profiling:
        phases["profiling"] = data.pop("profiling")
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
        data["datasets"] = {"default": data.pop("dataset")}

    if "phases" in data:
        phases = data["phases"]
        if isinstance(phases, dict) and "type" in phases:
            data["phases"] = {"default": phases}


def normalize_benchmark_input(data: Any) -> Any:
    """Normalize BenchmarkConfig input before Pydantic validation.

    Handles:
        - model -> models (singular to plural)
        - dataset -> datasets (singular to plural, wrapped with key 'default')
        - phases: single config (has 'type') -> {'default': config}
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
    is_composed = "source" in config and "augment" in config
    if is_composed:
        return
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


def _normalize_single_dataset(name: str, config: Any, dataset_types: tuple) -> Any:
    # Accept already-constructed Pydantic models (for programmatic use)
    if isinstance(config, dataset_types):
        return config
    if not isinstance(config, dict):
        raise ValueError(
            f"Dataset '{name}' configuration must be a dictionary or Pydantic model"
        )

    _hoist_synthetic_prompt_fields(config)

    is_composed = "source" in config and "augment" in config and "type" not in config
    if is_composed:
        return config
    if "type" not in config:
        config["type"] = "synthetic"
    return config


def parse_datasets_input(v: Any) -> dict[str, Any]:
    """Parse dataset configurations, handling composed datasets.

    Composed datasets don't have a 'type' field but have 'source' and
    'augment'. Accepts already-constructed Pydantic models for
    programmatic use.
    """
    from aiperf.config.dataset import (
        ComposedDataset,
        FileDataset,
        PublicDataset,
        SyntheticDataset,
    )

    dataset_types = (SyntheticDataset, FileDataset, PublicDataset, ComposedDataset)

    if not isinstance(v, dict):
        raise ValueError("datasets must be a dictionary")

    return {
        name: _normalize_single_dataset(name, cfg, dataset_types)
        for name, cfg in v.items()
    }
