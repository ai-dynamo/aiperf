# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for config-file resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


def promote_benchmark_magic_lists(
    merged: dict[str, Any],
    cli_config: CLIConfig,
    *,
    promote_cli_dataset_magic_lists: Any,
    promote_magic_lists_to_sweep_block: Any,
    retarget_dataset_magic_lists: Any,
) -> None:
    benchmark = merged.get("benchmark")
    if not isinstance(benchmark, dict):
        return

    sweep_type = getattr(cli_config, "sweep_type", "grid")
    promote_cli_dataset_magic_lists(benchmark, cli_config, sweep_type=sweep_type)
    retarget_dataset_magic_lists(benchmark)
    promote_magic_lists_to_sweep_block(benchmark, sweep_type=sweep_type)
    promoted_sweep = benchmark.pop("sweep", None)
    if not isinstance(promoted_sweep, dict):
        return

    existing_sweep = merged.get("sweep")
    if isinstance(existing_sweep, dict):
        existing_sweep.setdefault("type", promoted_sweep.get("type", sweep_type))
        existing_sweep.setdefault("parameters", {})
        existing_sweep["parameters"].update(promoted_sweep.get("parameters", {}))
    else:
        merged["sweep"] = promoted_sweep
