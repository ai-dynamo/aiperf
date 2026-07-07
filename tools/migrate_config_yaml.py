#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan A migration script — re-indents pre-restructure flat YAML to envelope shape.

Hard-cut migration tool. Idempotent: running on already-migrated YAML is a no-op.

Usage:
    uv run python tools/migrate_config_yaml.py path/to/config.yaml --in-place
    uv run python tools/migrate_config_yaml.py path/to/config.yaml > new.yaml
    uv run python tools/migrate_config_yaml.py - < flat.yaml > envelope.yaml

Behavior:
- Body fields (the loader's `_BODY_KEYS`: model/models, endpoint,
  dataset/datasets, phases, artifacts, slos, tokenizer, gpu_telemetry,
  server_metrics (plus camelCase aliases), runtime, logging, metrics,
  accuracy) at the top level get re-indented under a `benchmark:` key.
- Singular top-level `dataset:` is promoted to `datasets: [...]`, mirroring
  the loader's auto-migration.
- Envelope fields ({sweep, multi_run, variables, random_seed}) stay at top.
- `sweep.parameters` (deprecated upstream spelling) is renamed to
  `sweep.variables` for grid/zip sweeps.
- `sweep.runs[i]` body keys get wrapped under `runs[i].benchmark`.
- `sweep.variables` keys (grid) gain `benchmark.` prefix unless they start
  with `benchmark.` or `variables.` already.
- Comments preserved via ruamel.yaml.
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

# Body fields that move under `benchmark:` in the new envelope shape.
# Imported from the loader so this tool can never drift from what
# `_auto_migrate_flat_shape` accepts (a drifted key would keep emitting the
# loader's deprecation warning forever after migration).
from aiperf.config.loader.core import _BODY_KEYS as BODY_KEYS

# Envelope fields that stay at top level. `benchmark` is the new wrapper key.
ENVELOPE_KEYS = frozenset(
    {"sweep", "multi_run", "variables", "random_seed", "benchmark"}
)

# Allowed grid sweep variable path prefixes.
GRID_PATH_PREFIXES = ("benchmark.", "variables.")


def _yaml() -> YAML:
    """Configure ruamel.yaml for round-trip preservation of comments and quoting."""
    yml = YAML()
    yml.preserve_quotes = True
    yml.indent(mapping=2, sequence=4, offset=2)
    return yml


def is_already_migrated(yaml_text: str) -> bool:
    """Return True if the YAML already uses the envelope shape.

    Heuristic: no top-level body keys present (i.e., the partition is clean).
    A document with only envelope keys (or empty) qualifies. A document with
    `benchmark:` at top level and no top-level body keys also qualifies.
    """
    yml = _yaml()
    data = yml.load(StringIO(yaml_text))
    if data is None:
        return True
    if not isinstance(data, dict):
        return True
    return BODY_KEYS.isdisjoint(data.keys())


def migrate_yaml_text(yaml_text: str) -> str:
    """Migrate a YAML string from flat shape to envelope shape, idempotent."""
    yml = _yaml()
    data = yml.load(StringIO(yaml_text))
    if data is None:
        return yaml_text
    if not isinstance(data, dict):
        return yaml_text
    _migrate_in_place(data)
    out = StringIO()
    yml.dump(data, out)
    return out.getvalue()


def _migrate_in_place(data: dict[str, Any]) -> None:
    """Mutate ``data`` from flat to envelope shape.

    Preserves source order: when body keys appear before envelope keys in the
    flat input, the new ``benchmark:`` block also lands before the envelope
    keys in the output. This keeps round-tripped diffs minimal.
    """
    keys = list(data.keys())
    body_keys_present = [k for k in keys if k in BODY_KEYS]
    if body_keys_present:
        first_body_idx = keys.index(body_keys_present[0])
        body_present: dict[str, Any] = {}
        for k in body_keys_present:
            value = data[k]
            del data[k]
            if k == "dataset":
                # Mirror the loader's promotion of the singular `dataset:`
                # shorthand to a one-element `datasets:` list. setdefault /
                # plain-assign ordering matches the loader: when both forms
                # are present the plural wins.
                if isinstance(value, dict) and "name" not in value:
                    value["name"] = "main"
                body_present.setdefault(
                    "datasets", value if isinstance(value, list) else [value]
                )
            else:
                body_present[k] = value
        # Merge into existing benchmark key if user partially migrated.
        if "benchmark" in data and isinstance(data["benchmark"], dict):
            for k, v in body_present.items():
                data["benchmark"][k] = v
        else:
            # Insert benchmark at the position where the first body key was.
            # ruamel.yaml CommentedMap supports insert(pos, key, value); fall
            # back to plain assignment for vanilla dicts.
            insert = getattr(data, "insert", None)
            if callable(insert):
                insert(first_body_idx, "benchmark", body_present)
            else:
                data["benchmark"] = body_present

    sweep = data.get("sweep")
    if isinstance(sweep, dict):
        rewrite_sweep_parameters_key(sweep)
        rewrite_grid_sweep_paths(sweep)
        runs = sweep.get("runs")
        if isinstance(runs, list):
            rewrite_scenario_runs(runs)


def rewrite_sweep_parameters_key(sweep: dict[str, Any]) -> None:
    """Rename the deprecated `sweep.parameters` key to `sweep.variables`.

    Upstream pre-restructure YAML spells the grid/zip variable map
    `parameters:`. The loader accepts it with a deprecation warning; this
    rewrite makes the migration permanent. When both spellings are present
    the file is left unchanged so the loader's targeted both-set error
    surfaces instead of this tool silently picking one.
    """
    if sweep.get("type", "grid") not in ("grid", "zip"):
        return
    if "parameters" not in sweep or "variables" in sweep:
        return
    keys = list(sweep.keys())
    pos = keys.index("parameters")
    value = sweep.pop("parameters")
    # ruamel.yaml CommentedMap supports insert(pos, key, value) to keep the
    # key at its original position; fall back to append for vanilla dicts.
    insert = getattr(sweep, "insert", None)
    if callable(insert):
        insert(pos, "variables", value)
    else:
        sweep["variables"] = value


def rewrite_grid_sweep_paths(sweep: dict[str, Any]) -> None:
    """Prefix grid `sweep.variables` path keys with `benchmark.` when needed.

    Only fires for grid sweeps (or sweeps without a ``type``). Keys already
    starting with ``benchmark.`` or ``variables.`` are left unchanged.
    """
    if sweep.get("type", "grid") != "grid":
        return
    variables = sweep.get("variables")
    if not isinstance(variables, dict):
        return
    rewritten: dict[str, Any] = {}
    for key, value in variables.items():
        if isinstance(key, str) and not key.startswith(GRID_PATH_PREFIXES):
            rewritten[f"benchmark.{key}"] = value
        else:
            rewritten[key] = value
    variables.clear()
    variables.update(rewritten)


def rewrite_scenario_runs(runs: list[dict[str, Any]]) -> None:
    """Wrap body fields inside scenario runs under a ``benchmark:`` key per run.

    Allowed top-level keys inside a run: ``name``, ``variables``, ``benchmark``.
    Anything else gets moved under ``run["benchmark"]``.
    """
    for run in runs:
        if not isinstance(run, dict):
            continue
        body_present = {k: run[k] for k in list(run.keys()) if k in BODY_KEYS}
        if not body_present:
            continue
        for k in body_present:
            del run[k]
        if "benchmark" in run and isinstance(run["benchmark"], dict):
            for k, v in body_present.items():
                run["benchmark"][k] = v
        else:
            run["benchmark"] = body_present


def _migrate_file(path: Path, *, in_place: bool) -> None:
    """Migrate a YAML file. If in_place, overwrite; else write to stdout."""
    text = path.read_text(encoding="utf-8")
    new_text = migrate_yaml_text(text)
    if in_place:
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
    else:
        sys.stdout.write(new_text)


def _migrate_stdin() -> None:
    """Read flat YAML from stdin and write envelope YAML to stdout."""
    text = sys.stdin.read()
    sys.stdout.write(migrate_yaml_text(text))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(
        description="Migrate AIPerf YAML configs from flat shape to envelope shape."
    )
    parser.add_argument(
        "path",
        help="Path to YAML file, or '-' for stdin.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the file in place. Ignored when path is '-'.",
    )
    args = parser.parse_args(argv)

    if args.path == "-":
        _migrate_stdin()
        return 0

    path = Path(args.path)
    if not path.exists() or not path.is_file():
        sys.stderr.write(f"error: not a file: {path}\n")
        return 2

    _migrate_file(path, in_place=args.in_place)
    return 0


if __name__ == "__main__":
    sys.exit(main())
