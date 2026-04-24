# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI-to-config converter.

Converts a flat ``CLIModel`` into a nested ``AIPerfConfig`` dict, then
validates through Pydantic. No magic — just explicit field-by-field mapping.

Section-builders live in ``_cli_sections.py`` and dataset-builders in
``_cli_dataset.py`` to keep each function within the ergonomics line budget.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from aiperf.config._cli_dataset import build_dataset
from aiperf.config._cli_sections import (
    build_accuracy,
    build_artifacts,
    build_endpoint,
    build_gpu_telemetry,
    build_logging_runtime,
    build_models,
    build_multi_run,
    build_profiling,
    build_server_metrics,
    build_tokenizer,
    build_warmup,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from aiperf.config.config import AIPerfConfig


def _init_random_seed(cli: BaseModel) -> None:
    from aiperf.common import random_generator as rng
    from aiperf.common.exceptions import InvalidStateError

    with contextlib.suppress(InvalidStateError):
        rng.init(cli.random_seed)


def _assemble_optional(nested: dict[str, Any], cli: BaseModel, s: set[str]) -> None:
    if tok := build_tokenizer(cli, s):
        nested["tokenizer"] = tok
    if acc := build_accuracy(cli, s):
        nested["accuracy"] = acc
    if mr := build_multi_run(cli, s):
        nested["multi_run"] = mr
    if "random_seed" in s:
        nested["random_seed"] = cli.random_seed
    if cli.goodput:
        nested["slos"] = dict(cli.goodput)


def build_aiperf_config(cli: BaseModel) -> AIPerfConfig:
    """Build an AIPerfConfig from a parsed CLIModel instance."""
    from aiperf.config.config import AIPerfConfig

    s = cli.model_fields_set

    endpoint = build_endpoint(cli, s)
    models = build_models(cli, s)
    prof = build_profiling(cli, s)

    phases: dict[str, Any] = {}
    if (warmup := build_warmup(cli, s)) is not None:
        phases["warmup"] = warmup
    phases["profiling"] = prof

    ds = build_dataset(cli, s)

    _init_random_seed(cli)
    artifacts = build_artifacts(cli, s)
    gpu_telemetry = build_gpu_telemetry(cli)
    server_metrics = build_server_metrics(cli)
    logging_dict, runtime_dict = build_logging_runtime(cli, s)

    nested: dict[str, Any] = {
        "endpoint": endpoint,
        "models": models,
        "phases": phases,
        "datasets": {"main": ds},
        "artifacts": artifacts,
        "gpu_telemetry": gpu_telemetry,
        "server_metrics": server_metrics,
    }
    if logging_dict:
        nested["logging"] = logging_dict
    if runtime_dict:
        nested["runtime"] = runtime_dict

    _assemble_optional(nested, cli, s)

    return AIPerfConfig(**nested)
