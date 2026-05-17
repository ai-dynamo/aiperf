# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Models configuration

This module hosts the model-selection Pydantic configs (per-model override,
advanced model item, weighted-strategy validation). Other top-level config
sections live in sibling submodules to keep any one file under the
ergonomics file-size cap:

* :mod:`aiperf.config.tokenizer`         — tokenizer config
* :mod:`aiperf.config.logging`           — logging config
* :mod:`aiperf.config.slos`              — SLOs type alias
* :mod:`aiperf.config.runtime`           — runtime config
* :mod:`aiperf.config.comm.inputs`       — IPC/TCP/DualBind communication configs
* :mod:`aiperf.config.sweep.multi_run`   — multi-run trial mechanics + convergence
* :mod:`aiperf.config.accuracy`          — accuracy benchmarking config
"""

from __future__ import annotations

from aiperf.config._models_core import ModelItem, ModelsAdvanced, TokenizerOverride
from aiperf.config.tokenizer import TokenizerConfig

__all__ = [
    "ModelItem",
    "ModelsAdvanced",
    "TokenizerConfig",
    "TokenizerOverride",
]
