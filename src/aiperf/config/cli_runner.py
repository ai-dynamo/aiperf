# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Implementation of the `aiperf config` subcommands.

The thin command-definition module lives at `aiperf.cli_commands.config`; the
heavy logic (template scaffolding, sweep expansion, output formatting) is here
so it is only imported when the command actually runs. This mirrors the
domain-package layout used by `aiperf.plugin.cli` for `aiperf plugins`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal


def run_init(
    *,
    template: str | None,
    list_templates: bool,
    search: str | None,
    category: str | None,
    verbose: bool,
    model: str | None,
    url: str | None,
    output: Path | None,
) -> None:
    """Implement ``aiperf config init``: scaffold a config from a template."""
    from aiperf.config._cli_runner_templates import (
        build_overrides,
        handle_list,
        handle_search,
    )
    from aiperf.config.templates import (
        apply_overrides,
        load_template_content,
        strip_spdx_header,
    )

    if list_templates:
        handle_list(category=category, verbose=verbose)
        return

    if search is not None:
        handle_search(search, verbose=verbose)
        return

    if template is None:
        print(
            "Specify a template with --template, or run "
            "'aiperf config init --list' to see what is available.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        content = load_template_content(template)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    overrides = build_overrides(content, model=model, url=url)
    if overrides:
        content = apply_overrides(content, overrides)
    content = strip_spdx_header(content)

    if output is None:
        sys.stdout.write(content)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Wrote {template} template to {output}")


def run_validate(*, config_file: Path) -> None:
    """Implement ``aiperf config validate``: load+validate a config file."""
    from aiperf.config.loader import validate_config_file
    from aiperf.config.loader.errors import ConfigurationError

    try:
        warnings = validate_config_file(config_file)
    except ConfigurationError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    for w in warnings:
        print(f"Warning: {w}", file=sys.stderr)

    if warnings:
        plural = "s" if len(warnings) != 1 else ""
        print(
            f"Configuration valid with {len(warnings)} warning{plural}: {config_file}"
        )
    else:
        print(f"Configuration valid: {config_file}")


def run_expand(
    *,
    config_file: Path,
    full: bool,
    index: int | None,
    fmt: Literal["text", "yaml", "json"],
) -> None:
    """Implement ``aiperf config expand``: preview sweep variations from a plan."""
    from aiperf.config.loader import load_benchmark_plan
    from aiperf.config.loader.errors import ConfigurationError
    from aiperf.config.sweep import AdaptiveSearchSweep

    try:
        plan = load_benchmark_plan(config_file)
    except ConfigurationError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    sweep_type = type(plan.sweep).__name__ if plan.sweep is not None else "none"
    if isinstance(plan.sweep, AdaptiveSearchSweep):
        print(
            "adaptive_search sweeps choose variations dynamically at "
            "run time; no static expansion is available. Pass a grid / zip "
            "/ scenarios / sobol / latin_hypercube sweep to preview.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    entries = list(zip(plan.variations, plan.configs, strict=True))
    if index is not None:
        if index < 0 or index >= len(entries):
            print(
                f"Error: --index {index} out of range; sweep has "
                f"{len(entries)} variation(s).",
                file=sys.stderr,
            )
            raise SystemExit(1)
        entries = [entries[index]]
        full = True  # single-index inspection always shows the body

    if fmt == "json":
        _emit_json(entries, sweep_type=sweep_type, full=full)
    elif fmt == "yaml":
        _emit_yaml(entries, sweep_type=sweep_type, full=full)
    else:
        _emit_text(entries, sweep_type=sweep_type, total=len(plan.configs), full=full)


def _emit_text(entries, *, sweep_type: str, total: int, full: bool) -> None:
    if len(entries) == total:
        print(
            f"Sweep type: {sweep_type} ({total} variation{'s' if total != 1 else ''})"
        )
    else:
        print(f"Sweep type: {sweep_type} (showing {len(entries)} of {total})")
    print()
    for variation, body in entries:
        print(f"[{variation.index}] dir={variation.dir_name}  label={variation.label}")
        if full:
            import yaml

            body_dict = body.model_dump(mode="json", exclude_none=True)
            dumped = yaml.safe_dump(
                body_dict, sort_keys=False, default_flow_style=False
            )
            indented = "\n".join(
                "    " + line for line in dumped.rstrip("\n").splitlines()
            )
            print(indented)
            print()


def _emit_yaml(entries, *, sweep_type: str, full: bool) -> None:
    import yaml

    payload = {
        "sweep_type": sweep_type,
        "variations": [
            {
                "index": variation.index,
                "dir_name": variation.dir_name,
                "label": variation.label,
                "values": dict(variation.values),
                **(
                    {"benchmark": body.model_dump(mode="json", exclude_none=True)}
                    if full
                    else {}
                ),
            }
            for variation, body in entries
        ],
    }
    sys.stdout.write(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))


def _emit_json(entries, *, sweep_type: str, full: bool) -> None:
    import orjson

    payload = {
        "sweep_type": sweep_type,
        "variations": [
            {
                "index": variation.index,
                "dir_name": variation.dir_name,
                "label": variation.label,
                "values": dict(variation.values),
                **(
                    {"benchmark": body.model_dump(mode="json", exclude_none=True)}
                    if full
                    else {}
                ),
            }
            for variation, body in entries
        ],
    }
    sys.stdout.write(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2, default=str).decode()
    )
    sys.stdout.write("\n")
