# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep export glue: artifact write + post-process hook.

Two public entry points:

- :func:`export_sweep_aggregate` -- builds the AggregateResult, writes the
  JSON + CSV pair, and returns the AggregateResult so the caller can summarize
  it.
- :func:`run_post_process_hook` -- looks up the recipe-supplied handler and
  writes its derived artifact under ``sweep_aggregate/``. Quarantines handler
  failures via a ``post_process_errors.json`` sidecar so a buggy handler
  doesn't lose the run's primary outputs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger


async def export_sweep_aggregate(
    aggregate_result: Any,
    aggregate_dir: Path,
    logger: AIPerfLogger,
) -> None:
    """Build exporters and write the sweep aggregate JSON + CSV pair.

    Caller has already assembled the AggregateResult; we own only the
    exporter wiring and the user-visible log lines that name the written
    paths.
    """
    from aiperf.exporters.aggregate import (
        AggregateExporterConfig,
        AggregateSweepCsvExporter,
        AggregateSweepJsonExporter,
    )

    exporter_config = AggregateExporterConfig(
        result=aggregate_result, output_dir=aggregate_dir
    )
    json_exporter = AggregateSweepJsonExporter(exporter_config)
    csv_exporter = AggregateSweepCsvExporter(exporter_config)

    json_path, csv_path = await asyncio.gather(
        json_exporter.export(), csv_exporter.export()
    )
    logger.info(f"Sweep aggregate JSON written to: {json_path}")
    logger.info(f"Sweep aggregate CSV written to: {csv_path}")


def run_post_process_hook(
    post_spec: Any,
    sweep_dict: dict[str, Any],
    aggregate_dir: Path,
    logger: AIPerfLogger,
) -> None:
    """Run the recipe's post-process handler and write its artifact.

    Looks the handler up in the ``search_recipe_post_process`` plugin
    category, calls ``handler.process(sweep_dict, params)``, and serializes
    the returned dict to ``aggregate_dir / post_spec.output_filename`` as
    indented JSON.

    Failures are quarantined: the handler's exception is logged and a
    ``post_process_errors.json`` sidecar is written alongside the standard
    sweep aggregate, but the caller still returns the aggregate dir
    successfully — the standard artifacts are independent of the post-process
    step and shouldn't be lost when a handler bug surfaces.
    """
    import orjson

    from aiperf.common.finite import scrub_non_finite
    from aiperf.plugin.enums import PluginType
    from aiperf.plugin.plugins import get_class

    handler_name = post_spec.handler
    try:
        handler_cls = get_class(PluginType.SEARCH_RECIPE_POST_PROCESS, handler_name)
        handler = handler_cls()
        result_dict = handler.process(sweep_dict, dict(post_spec.params))
        out_path = aggregate_dir / post_spec.output_filename
        # orjson maps NaN/inf -> null, which collides with the SLABreachKnee
        # contract (`observed=None` means metric was missing; numeric means
        # present). A NaN observed value would round-trip indistinguishably
        # from "missing" and corrupt downstream tooling. Replace non-finite
        # floats with None before serialization so the contract holds across
        # every handler's artifact.
        result_dict = scrub_non_finite(result_dict)
        out_path.write_bytes(orjson.dumps(result_dict, option=orjson.OPT_INDENT_2))
        logger.info(f"Post-process artifact written: {out_path}")
    except Exception as exc:  # noqa: BLE001 - quarantine: see docstring
        errors_path = aggregate_dir / "post_process_errors.json"
        payload = {
            "handler": handler_name,
            "output_filename": post_spec.output_filename,
            "error": str(exc),
            "type": type(exc).__name__,
        }
        errors_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        logger.warning(
            f"Post-process handler {handler_name!r} failed: {exc}; "
            f"standard sweep aggregates unaffected (errors in {errors_path})."
        )
