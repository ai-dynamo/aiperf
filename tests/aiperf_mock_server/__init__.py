# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Mock Server - In-process testing utilities.

This package provides mock server infrastructure for testing AIPerf without
requiring actual HTTP servers.
"""

from .__main__ import main, serve
from .config import MockServerConfig

__all__ = ["MockServerConfig", "main", "serve"]
