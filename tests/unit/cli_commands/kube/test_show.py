# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf kube show — render AIPerfJob CR with Jinja2/env-vars resolved."""

from __future__ import annotations

from pathlib import Path

import yaml


def _minimal_cr() -> dict:
    """Minimal valid AIPerfJob CR dict."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": "test-job"},
        "spec": {
            "image": "nvcr.io/nvidia/aiperf:latest",
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": {
                    "main": {
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 32, "osl": 16},
                    }
                },
                "phases": {
                    "default": {"type": "concurrency", "requests": 10, "concurrency": 1}
                },
            },
        },
    }


def _write(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(doc, sort_keys=False))
    return path


def test_show_module_importable() -> None:
    """The show module must be importable and expose an `app` attribute."""
    from aiperf.cli_commands.kube import show

    assert hasattr(show, "app"), "show.app (cyclopts App) must be defined"


def test_show_registered_in_kube_app() -> None:
    """The `show` subcommand must be wired into `aiperf kube`."""
    from aiperf.cli_commands.kube._app import app

    # cyclopts App iteration yields registered command names as strings
    # (alongside flags like --help). We only care that "show" is registered.
    command_names = set(app)
    assert "show" in command_names
