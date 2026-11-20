# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for the end-to-end testing framework.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_repo_root() -> Path:
    """Get the repository root path from the current test location"""
    return Path(__file__).resolve().parents[3]


def setup_logging(
    level: int = logging.INFO, log_file: str = "test_execution.log"
) -> None:
    """Setup consistent logging configuration across all modules"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
