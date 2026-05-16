#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate config example YAML files.

This validator checks:
1. All src/aiperf/config/templates/*.yaml files can be parsed and loaded
2. YAML validates against JSON schema (structure validation)
3. Jinja2 templates render correctly
4. Sweep configurations expand properly
5. All Pydantic validation passes (semantic validation)

Usage:
    python -m tools.validate_config_examples
    python -m tools.validate_config_examples --verbose
    python -m tools.validate_config_examples --skip-schema  # Skip JSON schema validation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from tools._core import (
    GeneratorError,
    console,
    print_error,
    print_out_of_date,
    print_section,
    print_step,
    print_up_to_date,
)

# =============================================================================
# Paths
# =============================================================================

EXAMPLES_DIR = Path("src/aiperf/config/templates")
SCHEMA_FILE = Path("src/aiperf/config/schema/aiperf-config.schema.json")


# =============================================================================
# Validation Error
# =============================================================================


class ConfigValidationError(GeneratorError):
    """Error validating config example files."""


# =============================================================================
# JSON Schema Validation
# =============================================================================


def load_json_schema() -> dict[str, Any] | None:
    """Load the JSON schema for config validation.

    Returns:
        Schema dict if available, None if schema file doesn't exist.

    Raises:
        ConfigValidationError: If schema file exists but can't be loaded.
    """
    if not SCHEMA_FILE.exists():
        return None

    try:
        return json.loads(SCHEMA_FILE.read_text())
    except json.JSONDecodeError as e:
        raise ConfigValidationError(
            "Invalid JSON schema file",
            {"path": str(SCHEMA_FILE), "error": str(e)},
        ) from e


def validate_against_schema(
    yaml_path: Path,
    schema: dict[str, Any],
) -> list[str]:
    """Validate a YAML file against the JSON schema.

    Args:
        yaml_path: Path to the YAML file.
        schema: JSON schema to validate against.

    Returns:
        List of validation error messages (empty if valid).
    """
    try:
        import jsonschema
    except ImportError:
        # jsonschema not installed, skip schema validation
        return []

    # Load YAML content
    try:
        data = yaml.safe_load(yaml_path.read_text())
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]

    if data is None:
        return ["Empty configuration file"]

    # Validate against schema
    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(schema)

    for error in validator.iter_errors(data):
        # Format error path
        path = ".".join(str(p) for p in error.path) if error.path else "(root)"
        errors.append(f"{path}: {error.message}")

    return errors


# =============================================================================
# Validation
# =============================================================================


def validate_examples(
    verbose: bool = False,
    skip_schema: bool = False,
) -> tuple[int, int, int, float]:
    """Validate all example YAML files can be loaded.

    Args:
        verbose: If True, show detailed progress for each file.
        skip_schema: If True, skip JSON schema validation.

    Returns:
        Tuple of (failed_count, total_files, total_configs, elapsed_seconds).
    """
    start = time.perf_counter()

    # Import here to catch import errors gracefully
    try:
        from aiperf.config import load_config
    except ImportError as e:
        raise ConfigValidationError(
            "Failed to import config",
            {"error": str(e), "hint": "Run: uv pip install -e ."},
        ) from e

    if not EXAMPLES_DIR.exists():
        raise ConfigValidationError(
            "Examples directory not found",
            {"path": str(EXAMPLES_DIR)},
        )

    yaml_files = sorted(EXAMPLES_DIR.glob("*.yaml"))
    if not yaml_files:
        raise ConfigValidationError(
            "No YAML files found",
            {"directory": str(EXAMPLES_DIR)},
        )

    # Load JSON schema for validation (if available and not skipped)
    schema = None
    if not skip_schema:
        schema = load_json_schema()
        if schema and verbose:
            print_step("Using JSON schema validation")

    failed = 0
    total_configs = 0

    for yaml_path in yaml_files:
        file_start = time.perf_counter()
        schema_warnings: list[str] = []
        pydantic_errors: list[str] = []
        config_count = 0

        try:
            # Step 1: Validate against JSON schema (structure check)
            # Schema errors are warnings unless Pydantic also fails
            if schema:
                schema_warnings = validate_against_schema(yaml_path, schema)

            # Step 2: Load with Pydantic (authoritative semantic validation)
            try:
                load_config(yaml_path)
                config_count = 1
                total_configs += config_count

            except Exception as e:
                pydantic_errors.append(str(e))

            file_ms = (time.perf_counter() - file_start) * 1000

            # Pydantic errors are always failures
            # Schema-only errors are warnings (Pydantic is authoritative)
            if pydantic_errors:
                failed += 1
                if verbose:
                    print_out_of_date(f"{yaml_path.name} ({file_ms:.0f}ms)")
                    for err in pydantic_errors:
                        console.print(f"    [dim]•[/] [red]{err}[/]")
                    for warn in schema_warnings:
                        console.print(f"    [dim]•[/] [yellow]{warn}[/]")
                else:
                    print_out_of_date(f"{yaml_path.name}: {pydantic_errors[0]}")
            elif verbose:
                # Show schema warnings but don't fail
                suffix = ""
                if schema_warnings:
                    suffix = f" [yellow]({len(schema_warnings)} schema warnings)[/]"

                if config_count > 1:
                    print_up_to_date(
                        f"{yaml_path.name} ({config_count} configs, {file_ms:.0f}ms){suffix}"
                    )
                else:
                    print_up_to_date(f"{yaml_path.name} ({file_ms:.0f}ms){suffix}")

                # Show schema warning details in verbose mode
                for warn in schema_warnings:
                    console.print(f"    [dim]•[/] [yellow]{warn}[/]")

        except Exception as e:
            failed += 1
            file_ms = (time.perf_counter() - file_start) * 1000
            if verbose:
                print_out_of_date(f"{yaml_path.name} ({file_ms:.0f}ms)")
                console.print(f"    [dim]•[/] [red]{e}[/]")
            else:
                print_out_of_date(f"{yaml_path.name}: {e}")

    elapsed = time.perf_counter() - start
    return failed, len(yaml_files), total_configs, elapsed


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Validate config example YAML files."""
    parser = argparse.ArgumentParser(description="Validate config example YAML files")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress for each file",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip JSON schema validation (only run Pydantic validation)",
    )
    args = parser.parse_args()

    print_section("Config Examples Validation")

    try:
        failed, total_files, total_configs, elapsed = validate_examples(
            verbose=args.verbose,
            skip_schema=args.skip_schema,
        )
    except GeneratorError as e:
        print_error(e, verbose=args.verbose)
        return 1
    except Exception as e:
        print_error(e, verbose=True)
        return 1

    console.print()

    if failed:
        console.print(
            f"[bold red]✗[/] {failed}/{total_files} file(s) failed validation. "
            f"[dim]({elapsed:.2f}s)[/]"
        )
        return 1

    if args.verbose:
        console.print(
            f"[bold green]✓[/] Validated {total_files} files "
            f"({total_configs} configs). [dim]({elapsed:.2f}s)[/]"
        )
    else:
        # Compact output for non-verbose mode
        print_step(f"Validated {total_files} files ({total_configs} configs)")
        console.print()
        console.print(
            f"[bold green]✓[/] All config examples are valid. [dim]({elapsed:.2f}s)[/]"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
