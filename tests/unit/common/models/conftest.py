# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for model tests."""

import pytest


@pytest.fixture
def base_message_data():
    """Base message data template."""
    return {
        "service_id": "test-service",
        "request_ns": 1234567890,
    }


@pytest.fixture
def error_details():
    """Standard error details for testing."""
    return {
        "message": "Test error",
        "type": "TestError",
    }


@pytest.fixture
def process_records_result():
    """Minimal valid ProcessRecordsResult data."""
    return {
        "results": {
            "records": [],
            "completed": True,
            "start_ns": 0,
            "end_ns": 1000,
            "profile_results": {},
        },
        "errors": [],
    }
