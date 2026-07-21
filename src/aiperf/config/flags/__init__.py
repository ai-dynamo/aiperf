# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
config.flags - CLI-only input layer.

CLIConfig is the cyclopts-facing input DTO. It carries CLI flag annotations
(CLIParameter, Groups) and Pydantic field metadata (Field, BeforeValidator,
AfterValidator), but NO domain validators - AIPerfConfig is the single
validation gate.

Hard rules (enforced by code review + ``tools/check_ergonomics.py``
``pydantic-fields`` check, with ``INTENTIONAL_PYDANTIC_FIELDS_EXEMPTIONS``
whitelisting ``CLIConfig`` itself):

1. New CLI flags add as a top-level field on CLIConfig itself. NEVER add
   new nested classes. CLIConfig is fully flat - all modality fields are
   hoisted with their modality prefix (image_batch_size, audio_batch_size,
   video_batch_size, etc.) to disambiguate cross-modality collisions on
   `batch_size`, `format`, etc.
2. NO domain validators on CLIConfig. Validation lives on AIPerfConfig.
3. The converter (aiperf.config.flags.converter) is the only module outside
   cli_commands/ that may read CLIConfig attributes.

Each field is declared inline as
``Annotated[type, Field(...), CLIParameter(...)]`` - there is intentionally
no factory helper wrapping this. See ``docs/dev/patterns.md`` § "Adding a
New CLI Flag" for the recipe.

Anywhere downstream of cli_commands/, only AIPerfConfig / BenchmarkPlan /
BenchmarkRun flow.
"""

from aiperf.config.flags.cli_config import CLIConfig

__all__ = ["CLIConfig"]
