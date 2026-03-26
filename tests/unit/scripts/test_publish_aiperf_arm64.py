# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the local arm64 AIPerf publish script."""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "publish_aiperf_arm64.py"
)


def _load_module():
    assert SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}"

    spec = importlib.util.spec_from_file_location("publish_aiperf_arm64", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_tag_uses_timestamp_and_short_sha() -> None:
    """Build tags should match the existing k8s-arm64 timestamp style."""
    module = _load_module()

    tag = module.build_tag(dt.datetime(2026, 3, 26, 5, 7, 15), "1d7f82a87cafe123")

    assert tag == "k8s-arm64-20260326-050715-1d7f82a87"


def test_rewrite_image_refs_updates_only_target_repository(tmp_path: Path) -> None:
    """Only nvcr AIPerf refs should be rewritten."""
    module = _load_module()

    first = tmp_path / "first.yaml"
    first.write_text(
        'image: "nvcr.io/nvidian/dynamo-dev/aiperf:k8s-arm64-old"\n',
        encoding="utf-8",
    )
    second = tmp_path / "second.yaml"
    second.write_text(
        "image: nvcr.io/nvidian/dynamo-dev/aiperf:another-old\n"
        "sidecar: nvcr.io/nvidian/dynamo-dev/other:keep-me\n",
        encoding="utf-8",
    )

    changed_files = module.rewrite_image_refs(
        [first, second],
        repository="nvcr.io/nvidian/dynamo-dev/aiperf",
        new_tag="k8s-arm64-20260326-050715-1d7f82a87",
    )

    assert changed_files == [first, second]
    assert (
        first.read_text(encoding="utf-8")
        == 'image: "nvcr.io/nvidian/dynamo-dev/aiperf:k8s-arm64-20260326-050715-1d7f82a87"\n'
    )
    assert second.read_text(encoding="utf-8") == (
        "image: nvcr.io/nvidian/dynamo-dev/aiperf:k8s-arm64-20260326-050715-1d7f82a87\n"
        "sidecar: nvcr.io/nvidian/dynamo-dev/other:keep-me\n"
    )


def test_build_push_command_uses_buildx_arm64_push() -> None:
    """The publish pipeline should build and push linux/arm64 via buildx."""
    module = _load_module()

    command = module.build_push_command(
        image="nvcr.io/nvidian/dynamo-dev/aiperf:k8s-arm64-20260326-050715-1d7f82a87",
        dockerfile="Dockerfile",
        build_context=Path("/repo"),
    )

    assert command == [
        "docker",
        "buildx",
        "build",
        "--platform",
        "linux/arm64",
        "--push",
        "-t",
        "nvcr.io/nvidian/dynamo-dev/aiperf:k8s-arm64-20260326-050715-1d7f82a87",
        "-f",
        "Dockerfile",
        "/repo",
    ]


def test_main_dry_run_skips_buildx_and_auth_checks(monkeypatch, capsys) -> None:
    """Dry-run should print actions without requiring Docker or NVCR auth."""
    module = _load_module()

    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("prerequisite check should not run during dry-run")

    monkeypatch.setattr(module, "require_buildx", _unexpected_call)
    monkeypatch.setattr(module, "docker_config_has_registry_auth", _unexpected_call)
    monkeypatch.setattr(module, "git_short_sha", lambda: "1d7f82a87")

    exit_code = module.main(
        [
            "--dry-run",
            "--file",
            "dev/deploy/mock-250k-streaming.yaml",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert (
        "Publishing image: nvcr.io/nvidian/dynamo-dev/aiperf:k8s-arm64-" in captured.out
    )


@pytest.mark.parametrize(
    "raw_path",
    [
        pytest.param("dev/deploy/mock-250k-streaming.yaml", id="repo-relative"),
        pytest.param("./dev/deploy/mock-250k-streaming.yaml", id="dot-relative"),
    ],
)  # fmt: skip
def test_resolve_target_files_resolves_relative_to_project_root(raw_path: str) -> None:
    """Relative --file paths should resolve from the repo root."""
    module = _load_module()

    resolved = module.resolve_target_files([raw_path])

    assert resolved == [module.PROJECT_ROOT / "dev/deploy/mock-250k-streaming.yaml"]
