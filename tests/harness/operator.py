# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator test helpers — sample data builders for AIPerfJob specs and API responses.

These are pure functions with no pytest coupling. Conftest files wrap them
as fixtures; test files can also import them directly.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Kubernetes Resource Body Builders
# =============================================================================


def build_sample_body() -> dict[str, Any]:
    """Create a sample Kubernetes resource body for event testing."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {
            "name": "test-job",
            "namespace": "default",
            "uid": "abc-123",
            # 2024-04-25T17:02:03Z -> epoch 1714064523; fixed so epoch-keyed
            # run dirs under results_layout.run_dir are deterministic.
            "creationTimestamp": "2024-04-25T17:02:03Z",
        },
        "spec": {
            "image": "aiperf:test",
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"url": "http://localhost:8000"},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": {
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                },
            },
        },
    }


# =============================================================================
# AIPerfJob Spec Builders (nested benchmark format)
# =============================================================================


def build_minimal_aiperfjob_spec() -> dict[str, Any]:
    """Create a minimal AIPerfJob spec for testing (nested benchmark format)."""
    return {
        "benchmark": {
            "models": ["test-model"],
            "endpoint": {
                "urls": ["http://localhost:8000/v1/chat/completions"],
            },
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": {
                "type": "concurrency",
                "requests": 10,
                "concurrency": 1,
            },
        },
    }


def build_full_aiperfjob_spec() -> dict[str, Any]:
    """Create a complete AIPerfJob spec with all options (nested benchmark format)."""
    return {
        "image": "aiperf:test",
        "imagePullPolicy": "Always",
        "connectionsPerWorker": 100,
        "benchmark": {
            "models": ["gpt-4"],
            "endpoint": {
                "urls": ["http://api.example.com/v1/chat/completions"],
            },
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "requests": 50,
                    "concurrency": 500,
                    "exclude_from_results": True,
                },
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 1000,
                    "concurrency": 500,
                },
            ],
        },
        "podTemplate": {
            "nodeSelector": {"gpu": "true"},
            "tolerations": [
                {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
            ],
            "annotations": {"prometheus.io/scrape": "true"},
            "labels": {"team": "ml-platform"},
            "imagePullSecrets": [{"name": "my-registry-secret"}],
            "serviceAccountName": "aiperf-sa",
            "env": [
                {"name": "DEBUG", "value": "true"},
                {
                    "name": "API_KEY",
                    "valueFrom": {
                        "secretKeyRef": {"name": "api-secrets", "key": "api-key"}
                    },
                },
            ],
            "volumes": [
                {"name": "creds", "secret": {"secretName": "my-creds"}},
            ],
            "volumeMounts": [
                {"name": "creds", "mountPath": "/etc/creds"},
            ],
        },
    }


def build_high_concurrency_spec() -> dict[str, Any]:
    """Create an AIPerfJob spec with high concurrency for worker scaling tests."""
    return {
        "connectionsPerWorker": 100,
        "benchmark": {
            "models": ["test-model"],
            "endpoint": {
                "urls": ["http://localhost:8000/v1/chat/completions"],
            },
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": {
                "type": "concurrency",
                "requests": 1000,
                "concurrency": 1000,
            },
        },
    }


# =============================================================================
# Progress API Response Builders
# =============================================================================


def build_progress_response_running() -> dict[str, Any]:
    """Create a progress API response for a running job."""
    return {
        "phases": {
            "warmup": {
                "phase": "warmup",
                "start_ns": 1000,
                "requests_completed": 50,
                "total_expected_requests": 50,
                "requests_per_second": 10.5,
                "requests_progress_percent": 100.0,
                "is_requests_complete": True,
            },
            "profiling": {
                "phase": "profiling",
                "start_ns": 2000,
                "requests_completed": 250,
                "total_expected_requests": 1000,
                "requests_per_second": 25.0,
                "requests_progress_percent": 25.0,
                "is_requests_complete": False,
                "requests_eta_sec": 30.0,
            },
        },
    }


def build_progress_response_with_error() -> dict[str, Any]:
    """Create a progress API response with an error."""
    return {
        "error": "Connection refused to endpoint",
        "phases": {
            "profiling": {
                "phase": "profiling",
                "start_ns": 1000,
                "requests_completed": 100,
                "total_expected_requests": 1000,
                "requests_per_second": 0.0,
                "requests_progress_percent": 10.0,
                "is_requests_complete": False,
            },
        },
    }


# =============================================================================
# Condition Builders
# =============================================================================


def build_sample_conditions_list() -> list[dict[str, Any]]:
    """Create a sample conditions list from Kubernetes status."""
    return [
        {
            "type": "ConfigValid",
            "status": "True",
            "reason": "ConfigParsed",
            "message": "Config is valid",
            "lastTransitionTime": "2026-01-15T10:00:00Z",
        },
        {
            "type": "ResourcesCreated",
            "status": "True",
            "reason": "Created",
            "message": "Resources created successfully",
            "lastTransitionTime": "2026-01-15T10:00:05Z",
        },
        {
            "type": "WorkersReady",
            "status": "False",
            "reason": "WorkersStarting",
            "message": "2/5 workers ready",
            "lastTransitionTime": "2026-01-15T10:00:10Z",
        },
    ]
