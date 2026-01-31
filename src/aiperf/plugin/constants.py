# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Final

# Supported schema versions for registry manifests
SUPPORTED_SCHEMA_VERSIONS: Final[tuple[str, ...]] = ("1.0",)

# Entry point group for package discovery
DEFAULT_ENTRY_POINT_GROUP: Final[str] = "aiperf.plugins"
