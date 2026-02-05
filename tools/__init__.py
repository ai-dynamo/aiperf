# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf code generation and documentation tools.

This package contains tools for generating:
- Plugin system artifacts (schemas, enums, overloads)
- CLI documentation
- Environment variable documentation

Usage:
    ./tools/generate_plugin_artifacts.py
    ./tools/generate_cli_docs.py
    ./tools/generate_env_vars_docs.py
"""

from tools._core import (
    # Constants
    CONSTRAINT_SYMBOLS,
    GENERATED_FILE_HEADER,
    SPDX_HEADER_MD,
    # Error classes
    CLIExtractionError,
    EnumGenerationError,
    GeneratedFile,
    # Generator infrastructure
    Generator,
    GeneratorError,
    GeneratorResult,
    OverloadGenerationError,
    ParseError,
    SchemaGenerationError,
    YAMLLoadError,
    # Console output
    console,
    error_console,
    main,
    make_generated_header,
    # Text utilities
    normalize_text,
    print_error,
    print_generated,
    print_out_of_date,
    print_section,
    print_step,
    print_up_to_date,
    print_updated,
    print_warning,
    run,
    # File utilities
    write_if_changed,
)

__all__ = [
    # Constants
    "CONSTRAINT_SYMBOLS",
    "GENERATED_FILE_HEADER",
    "SPDX_HEADER_MD",
    "make_generated_header",
    # Error classes
    "CLIExtractionError",
    "EnumGenerationError",
    "GeneratorError",
    "OverloadGenerationError",
    "ParseError",
    "SchemaGenerationError",
    "YAMLLoadError",
    # Console output
    "console",
    "error_console",
    "print_error",
    "print_generated",
    "print_out_of_date",
    "print_section",
    "print_step",
    "print_up_to_date",
    "print_updated",
    "print_warning",
    # File utilities
    "write_if_changed",
    # Text utilities
    "normalize_text",
    # Generator infrastructure
    "Generator",
    "GeneratedFile",
    "GeneratorResult",
    "main",
    "run",
]
