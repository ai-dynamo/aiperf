# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load configs and generate the config schema reference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import orjson

from aiperf.dataset.agentic_code_gen.models import SessionDistributionConfig

CONFIG_SCHEMA_ID = (
    "https://ai-dynamo.github.io/aiperf/schemas/agentic-code-gen-config.json"
)
CONFIG_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

_CONFIGS_DIR = Path(__file__).parent / "configs"
_SPEC_CONFIG_NAME = "spec"
DEFAULT_SCHEMA_PATH = _CONFIGS_DIR / f"{_SPEC_CONFIG_NAME}.json"

_PERMISSIVE_CONFIG_NOTE = (
    "This schema is a reference for recognized Agentic Code generator config "
    "fields. AIPerf config parsing is permissive for forward compatibility: "
    "unknown fields are accepted by the base model, but only the documented "
    "properties below are interpreted by the generator."
)


def list_bundled_configs() -> list[str]:
    """Return names of bundled config files (without .json extension)."""
    return sorted(
        p.stem for p in _CONFIGS_DIR.glob("*.json") if p.stem != _SPEC_CONFIG_NAME
    )


def load_config(path_or_name: str) -> SessionDistributionConfig:
    """Load a config from a bundled name or file path.

    Resolution order:
    1. If *path_or_name* matches a bundled config name, load it.
    2. If it is a path to an existing file, load it.
    3. Otherwise raise FileNotFoundError.

    Supports both raw config JSON and manifest.json (has generation_params wrapper).
    """
    bundled = _CONFIGS_DIR / f"{path_or_name}.json"
    if bundled.is_file():
        if bundled.stem == _SPEC_CONFIG_NAME:
            raise ValueError(
                "spec.json is a JSON Schema reference, not a loadable config"
            )
        return _load_json(bundled)

    path = Path(path_or_name)
    if path.is_file():
        if path.resolve() == (_CONFIGS_DIR / f"{_SPEC_CONFIG_NAME}.json").resolve():
            raise ValueError(
                "spec.json is a JSON Schema reference, not a loadable config"
            )
        return _load_json(path)

    available = list_bundled_configs()
    raise FileNotFoundError(
        f"Config '{path_or_name}' not found. "
        f"Bundled configs: {available}. Or provide a file path."
    )


def _load_json(path: Path) -> SessionDistributionConfig:
    data = orjson.loads(path.read_bytes())
    if not isinstance(data, dict):
        raise ValueError(f"Config '{path}' must be a JSON object")
    if data.get("$id") == CONFIG_SCHEMA_ID and "properties" in data:
        raise ValueError("spec.json is a JSON Schema reference, not a loadable config")
    if "generation_params" in data:
        data = data["generation_params"]
        if not isinstance(data, dict):
            raise ValueError(f"Config '{path}' generation_params must be a JSON object")
    return SessionDistributionConfig(**data)


def build_config_schema() -> dict[str, Any]:
    """Build the JSON Schema dictionary for the public config reference."""
    schema = SessionDistributionConfig.model_json_schema()
    description = schema.get("description", "").strip()
    schema["description"] = (
        f"{description}\n\n{_PERMISSIVE_CONFIG_NOTE}"
        if description
        else _PERMISSIVE_CONFIG_NOTE
    )
    return {
        "$schema": CONFIG_SCHEMA_DRAFT,
        "$id": CONFIG_SCHEMA_ID,
        **schema,
    }


def dump_config_schema() -> bytes:
    """Serialize the generated config schema with stable formatting."""
    return (
        orjson.dumps(
            build_config_schema(),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        + b"\n"
    )


def write_config_schema(path: Path = DEFAULT_SCHEMA_PATH) -> Path:
    """Write the generated config schema and return the output path."""
    path.write_bytes(dump_config_schema())
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate agentic-code-gen config spec.json"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the output file does not match the generated schema",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to write or check",
    )
    args = parser.parse_args(argv)

    expected = dump_config_schema()
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != expected:
            print(f"{args.output} is out of date")
            return 1
        print(f"{args.output} is up to date")
        return 0

    args.output.write_bytes(expected)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
