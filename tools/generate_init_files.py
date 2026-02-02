#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate __init__.py files using mkinit.

Usage:
    ./tools/generate_init_files.py
    ./tools/generate_init_files.py --check
    ./tools/generate_init_files.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution: add repo root to path for 'tools' package imports
if __name__ == "__main__" and "tools" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import contextlib
import subprocess

from tools._core import (
    GeneratedFile,
    Generator,
    GeneratorError,
    GeneratorResult,
    main,
    print_step,
)

# =============================================================================
# Configuration
# =============================================================================

SRC_DIR = Path("src/aiperf")


# =============================================================================
# Errors
# =============================================================================


class InitFileGenerationError(GeneratorError):
    """Error during __init__.py generation."""

    pass


# =============================================================================
# Generator
# =============================================================================


class InitFilesGenerator(Generator):
    """Generator for __init__.py files using mkinit."""

    name = "Init Files"
    description = "__init__.py files"

    def generate(self) -> GeneratorResult:
        """Generate __init__.py files."""
        if not SRC_DIR.exists():
            raise InitFileGenerationError(
                f"Source directory not found: {SRC_DIR}",
                {"hint": "Run from the repository root directory"},
            )

        # Find all existing __init__.py files and their current content
        init_files = list(SRC_DIR.rglob("__init__.py"))
        old_contents = {f: f.read_text() for f in init_files}

        if self.verbose:
            print_step(f"Found {len(init_files)} existing __init__.py files")

        # Run mkinit to generate/update __init__.py files
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mkinit",
                    "--write",
                    "--black",
                    "--nomods",
                    "--recursive",
                    str(SRC_DIR),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            if self.verbose and result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        print_step(line)
        except subprocess.CalledProcessError as e:
            raise InitFileGenerationError(
                "mkinit failed",
                {"stderr": e.stderr, "returncode": e.returncode},
            ) from e
        except FileNotFoundError as e:
            raise InitFileGenerationError(
                "mkinit not found",
                {"hint": "Run: uv pip install mkinit"},
            ) from e

        # Run ruff to fix imports (mkinit sometimes doesn't sort correctly)
        init_files_after = list(SRC_DIR.rglob("__init__.py"))
        if init_files_after:
            with contextlib.suppress(FileNotFoundError):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ruff",
                        "check",
                        "--fix",
                        "--select=I",  # Only import sorting
                        "--quiet",
                        *[str(f) for f in init_files_after],
                    ],
                    capture_output=True,
                    text=True,
                    check=False,  # ruff returns non-zero if it makes changes
                )

        # Compare old and new contents to find changed files
        files: list[GeneratedFile] = []
        for init_file in init_files_after:
            new_content = init_file.read_text()
            old_content = old_contents.get(init_file, "")
            if new_content != old_content:
                files.append(GeneratedFile(init_file, new_content))

        return GeneratorResult(
            files=files,
            summary=f"[bold]{len(init_files_after)}[/] __init__.py files",
        )


if __name__ == "__main__":
    main(InitFilesGenerator)
