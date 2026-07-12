# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf command namespace for every DynoSim and Mocker product surface.

The run and live-mocker commands deliberately forward the raw argument vector
to Dynamo's canonical parsers. AIPerf therefore does not maintain a second,
drift-prone copy of either CLI schema: a new DynoSim or Mocker flag is usable
through this namespace immediately and is validated by the owning parser.
"""

from __future__ import annotations

import importlib
import importlib.resources
import json
import math
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from cyclopts import App, Parameter

app = App(
    name="dynosim",
    help="Run DynoSim trials and sweeps or launch live Mocker workers.",
)

_RAW_ARGUMENTS = Parameter(
    allow_leading_hyphen=True,
    consume_multiple=True,
    help="Arguments forwarded unchanged to the canonical Dynamo command.",
)


class SweepOperation(str, Enum):
    """Public replay-optimization operation selected by ``dynosim sweep``."""

    AGG = "agg"
    DISAGG = "disagg"
    COMPARE_TOPOLOGIES = "compare_topologies"
    COMPARE_AIC = "compare_aic"


def _import_symbol(module_name: str, symbol: str) -> Any:
    """Import one optional Dynamo symbol with an actionable dependency error."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "DynoSim support requires an ai-dynamo installation containing "
            f"{module_name!r}; install Dynamo in the AIPerf environment"
        ) from error
    return getattr(module, symbol)


@contextmanager
def _temporary_argv(program: str, arguments: Sequence[str]) -> Iterator[None]:
    """Temporarily install an argv for canonical commands that own argparse."""
    previous = sys.argv
    sys.argv = [program, *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def _run_argparse_main(
    main: Callable[..., int | None],
    arguments: Sequence[str],
    *,
    accepts_argv: bool,
    program: str,
) -> None:
    """Run a canonical argparse entry point without altering its semantics."""
    if accepts_argv:
        status = main(list(arguments))
    else:
        with _temporary_argv(program, arguments):
            status = main()
    if status not in (None, 0):
        raise SystemExit(status)


@app.command(name="run")
def run(
    arguments: Annotated[list[str], _RAW_ARGUMENTS],
) -> None:
    """Run one canonical DynoSim replay through the AIPerf command frontend.

    Every argument after ``run`` is passed byte-for-byte to
    ``python -m dynamo.replay``. This includes offline/online execution,
    aggregate/disaggregate topology, all trace formats, router/AIC/planner
    controls, per-request artifacts, simulation caps, and goodput SLAs.
    """
    main = _import_symbol("dynamo.replay.main", "main")
    _run_argparse_main(
        main,
        arguments,
        accepts_argv=True,
        program="aiperf dynosim run",
    )


@app.command(name="mocker")
def mocker(
    arguments: Annotated[list[str], _RAW_ARGUMENTS],
) -> None:
    """Launch canonical live Mocker workers through the AIPerf frontend.

    Every argument after ``mocker`` is passed unchanged to
    ``python -m dynamo.mocker``. The owning parser therefore retains authority
    over engine, scheduler, KV tier, discovery, request/event transport, ZMQ,
    bootstrap, AIC, reasoning, and multi-worker behavior.
    """
    main = _import_symbol("dynamo.mocker.main", "main")
    _run_argparse_main(
        main,
        arguments,
        accepts_argv=False,
        program="aiperf dynosim mocker",
    )


def _load_structured(path: Path) -> dict[str, Any]:
    """Load a replay-optimization spec from JSON or YAML."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            yaml = importlib.import_module("yaml")
        except ModuleNotFoundError as error:
            raise RuntimeError("YAML sweep specs require PyYAML") from error
        value = yaml.safe_load(text)
    else:
        raise ValueError("DynoSim sweep spec must end in .json, .yaml, or .yml")
    if not isinstance(value, dict):
        raise ValueError("DynoSim sweep spec must contain an object at its root")
    return value


def _json_safe(value: Any) -> Any:
    """Convert Pydantic/pandas/numpy results into strict JSON-compatible data."""
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if hasattr(value, "to_dict") and value.__class__.__module__.startswith("pandas"):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _result_payload(operation: SweepOperation, result: Any) -> dict[str, Any]:
    """Normalize every public sweep operation to one stable artifact envelope."""
    if operation in {SweepOperation.AGG, SweepOperation.DISAGG}:
        return {
            "operation": operation.value,
            "best_feasible": _json_safe(result.best_feasible),
            "best_infeasible": _json_safe(result.best_infeasible),
            "evaluated": _json_safe(result.evaluated_df),
            "feasible": _json_safe(result.feasible_df),
        }
    if operation is SweepOperation.COMPARE_TOPOLOGIES:
        return {
            "operation": operation.value,
            "chosen_mode": result["chosen_mode"],
            "chosen_best": _json_safe(result["chosen_best"]),
            "agg": _result_payload(SweepOperation.AGG, result["agg_result"]),
            "disagg": _result_payload(
                SweepOperation.DISAGG, result["disagg_result"]
            ),
        }
    replay = result["replay_result"]
    return {
        "operation": operation.value,
        "aic_best": _json_safe(result["aic_best"]),
        "aic_pareto": _json_safe(result["aic_pareto_df"]),
        "replay_best": _json_safe(result["replay_best"]),
        "replay": _result_payload(SweepOperation.DISAGG, replay),
    }


def _capability_manifest() -> dict[str, Any]:
    """Load the checked DynoSim/Mocker requirement inventory."""
    resource = importlib.resources.files("aiperf").joinpath(
        "dynosim_capabilities.json"
    )
    value = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("DynoSim capability manifest root is not an object")
    return value


@app.command(name="capabilities")
def capabilities(
    *,
    output: Annotated[
        Path | None,
        Parameter(name="--output", help="Manifest JSON path; stdout when omitted."),
    ] = None,
) -> None:
    """Print the machine-checked DynoSim/Mocker capability manifest."""
    rendered = json.dumps(
        _capability_manifest(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


@app.command(name="sweep")
def sweep(
    spec_file: Annotated[
        Path,
        Parameter(help="ReplayOptimizeSpec JSON or YAML file."),
    ],
    *,
    operation: Annotated[
        SweepOperation,
        Parameter(name="--operation", help="Sweep/search operation to run."),
    ] = SweepOperation.DISAGG,
    output: Annotated[
        Path | None,
        Parameter(name="--output", help="Result JSON path; stdout when omitted."),
    ] = None,
) -> None:
    """Run any public DynoSim replay-optimization workflow.

    The input is the canonical ``ReplayOptimizeSpec`` schema, including full
    engine/router dictionaries, hardware budget, synthetic or trace workload,
    SLA bounds, objective, and process-level parallelism. Four operations are
    exposed: aggregate search, disaggregate search, topology comparison, and
    AIC-versus-replay comparison.
    """
    _import_symbol(
        "dynamo.profiler.utils.replay_optimize", "ReplayOptimizeSpec"
    )
    module = importlib.import_module("dynamo.profiler.utils.replay_optimize")
    spec_type = module.ReplayOptimizeSpec
    spec = spec_type.model_validate(_load_structured(spec_file))
    operations: dict[SweepOperation, Callable[[Any], Any]] = {
        SweepOperation.AGG: module.optimize_dense_agg_with_replay,
        SweepOperation.DISAGG: module.optimize_dense_disagg_with_replay,
        SweepOperation.COMPARE_TOPOLOGIES: module.compare_agg_and_disagg_with_replay,
        SweepOperation.COMPARE_AIC: module.compare_aic_and_replay_disagg,
    }
    payload = _result_payload(operation, operations[operation](spec))
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
