# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-image release input tests for native runner packaging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import runner_release_input

_REVISION = "a" * 40


def _binary(tmp_path: Path) -> Path:
    binary = tmp_path / "aiperf-runner"
    binary.write_bytes(b"\x7fELFtest-runner-image")
    binary.chmod(0o755)
    return binary


def _capabilities(binary: Path, *, offline: bool) -> dict[str, object]:
    pairs = (
        [["dynamo_offline", "graph"], ["dynamo_offline", "scheduled"]]
        if offline
        else []
    )
    transports = (
        [{"id": "dynamo_offline"}, {"id": "http"}]
        if offline
        else [{"id": "http"}]
    )
    return {
        "event": "runner_capabilities",
        "distribution_id": runner_release_input._distribution_id(binary),
        "supported_pairs": pairs,
        "transports": transports,
    }


def _write_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    offline: bool,
) -> tuple[Path, Path, Path]:
    binary = _binary(tmp_path)
    cargo_lock = tmp_path / "Cargo.lock"
    cargo_lock.write_text("version = 4\n")
    monkeypatch.setattr(
        runner_release_input,
        "_load_capabilities",
        lambda _binary: _capabilities(binary, offline=offline),
    )
    manifest = runner_release_input.create_manifest(
        binary=binary,
        cargo_lock=cargo_lock,
        source_revision=_REVISION,
        features=["dynamo-offline"] if offline else [],
        dependency_revisions=({"dynamo-aiperf-native": "b" * 40} if offline else {}),
    )
    manifest_path = tmp_path / "runner-build.json"
    manifest_path.write_text(json.dumps(manifest))
    return binary, cargo_lock, manifest_path


@pytest.mark.parametrize("profile", ["online", "offline"])
def test_release_input_binds_exact_image_lock_and_capability_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: runner_release_input.RunnerProfile,
) -> None:
    offline = profile == "offline"
    binary, cargo_lock, manifest_path = _write_manifest(
        monkeypatch,
        tmp_path,
        offline=offline,
    )

    verified = runner_release_input.verify_release_input(
        binary=binary,
        manifest_path=manifest_path,
        cargo_lock=cargo_lock,
        source_revision=_REVISION,
        profile=profile,
    )

    assert verified["profile"] == profile
    assert verified["features"] == (["dynamo-offline"] if offline else [])
    assert verified["dependency_revisions"] == (
        {"dynamo-aiperf-native": "b" * 40} if offline else {}
    )
    assert verified["distribution_id"] == runner_release_input._distribution_id(binary)


def test_release_input_rejects_binary_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary, cargo_lock, manifest_path = _write_manifest(
        monkeypatch,
        tmp_path,
        offline=False,
    )
    binary.write_bytes(b"\x7fELFreplaced-runner-image")

    with pytest.raises(RuntimeError, match="exact binary"):
        runner_release_input.verify_release_input(
            binary=binary,
            manifest_path=manifest_path,
            cargo_lock=cargo_lock,
            source_revision=_REVISION,
            profile="online",
        )


def test_online_profile_rejects_offline_capability_or_feature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary, cargo_lock, manifest_path = _write_manifest(
        monkeypatch,
        tmp_path,
        offline=True,
    )

    with pytest.raises(RuntimeError, match="online runner manifest"):
        runner_release_input.verify_release_input(
            binary=binary,
            manifest_path=manifest_path,
            cargo_lock=cargo_lock,
            source_revision=_REVISION,
            profile="online",
        )


def test_offline_profile_requires_both_executable_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary, cargo_lock, manifest_path = _write_manifest(
        monkeypatch,
        tmp_path,
        offline=True,
    )
    capabilities = _capabilities(binary, offline=True)
    capabilities["supported_pairs"] = [["dynamo_offline", "scheduled"]]
    monkeypatch.setattr(
        runner_release_input,
        "_load_capabilities",
        lambda _binary: capabilities,
    )

    with pytest.raises(RuntimeError, match="scheduled and graph pairs"):
        runner_release_input.verify_release_input(
            binary=binary,
            manifest_path=manifest_path,
            cargo_lock=cargo_lock,
            source_revision=_REVISION,
            profile="offline",
        )
