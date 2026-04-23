# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helper for loading the dataset manager's tokenizer off the event loop."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiperf.common.tokenizer import Tokenizer

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


async def load_tokenizer_for_run(run: BenchmarkRun) -> Tokenizer:
    """Load the tokenizer selected by resolver chain or falling back to the config.

    Exceptions propagate so controller_utils can display the error panel.
    """
    tokenizer_config = run.cfg.tokenizer
    resolved_names = run.resolved.tokenizer_names
    model_name = run.cfg.get_model_names()[0]

    if resolved_names and model_name in resolved_names:
        tokenizer_name = resolved_names[model_name]
        resolve_alias = False
    else:
        tokenizer_name = (
            tokenizer_config.name if tokenizer_config else None
        ) or model_name
        resolve_alias = True

    return await asyncio.to_thread(
        Tokenizer.from_pretrained,
        tokenizer_name,
        trust_remote_code=tokenizer_config.trust_remote_code
        if tokenizer_config
        else False,
        revision=tokenizer_config.revision if tokenizer_config else "main",
        resolve_alias=resolve_alias,
    )
