# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

# The modules in this package use bare imports (from data_types import ...)
# because they are also run standalone from their own directory.
# Add this directory to sys.path so those imports resolve when the package
# is imported from outside (e.g., from tests/unit/ci/).
_pkg_dir = str(Path(__file__).parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
