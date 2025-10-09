# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for records tests."""

# Import shared constants and fixtures from root conftest
from tests.conftest import (  # noqa: F401
    DEFAULT_FIRST_RESPONSE_NS,
    DEFAULT_INPUT_TOKENS,
    DEFAULT_LAST_RESPONSE_NS,
    DEFAULT_OUTPUT_TOKENS,
    DEFAULT_START_TIME_NS,
    sample_parsed_record,
    sample_request_record,
)
