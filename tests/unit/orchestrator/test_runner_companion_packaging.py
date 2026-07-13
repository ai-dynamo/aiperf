# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static release-contract checks for the native runner companion."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tomllib
import types
from pathlib import Path

import pytest

from tools.stage_stock_evaluator_roots import (
    PROVIDER_ROOT_SPECS,
    PROVIDER_ROOTS_REGISTRY,
    PROVIDER_ROOTS_SCHEMA,
)

_ROOT = Path(__file__).resolve().parents[3]


def _toml(relative: str) -> dict:
    return tomllib.loads((_ROOT / relative).read_text())


def _build_hook(monkeypatch: pytest.MonkeyPatch):
    interface = types.ModuleType("hatchling.builders.hooks.plugin.interface")
    interface.BuildHookInterface = object
    modules = {
        "hatchling": types.ModuleType("hatchling"),
        "hatchling.builders": types.ModuleType("hatchling.builders"),
        "hatchling.builders.hooks": types.ModuleType("hatchling.builders.hooks"),
        "hatchling.builders.hooks.plugin": types.ModuleType(
            "hatchling.builders.hooks.plugin"
        ),
        "hatchling.builders.hooks.plugin.interface": interface,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = _ROOT / "packaging/aiperf-runner/hatch_build.py"
    spec = importlib.util.spec_from_file_location("aiperf_runner_hatch_build", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _staged_roots(module, root: Path) -> None:
    entries = []
    for root_id, kind, relative in module._PROVIDER_ROOT_SPECS:
        content = f"{root_id}\n".encode()
        target = root / relative / "proof.txt"
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
        members = {"proof.txt": module.hashlib.sha256(content).hexdigest()}
        entries.append(
            {
                "file_count": 1,
                "id": root_id,
                "kind": kind,
                "path": relative,
                "tree_sha256": module._tree_sha256(members),
            }
        )
    registry = {
        "platform": "linux-x86_64",
        "roots": entries,
        "schema_version": module._PROVIDER_ROOTS_SCHEMA,
    }
    (root / module._PROVIDER_ROOTS_REGISTRY).write_bytes(
        module._canonical_registry(registry)
    )


def _runner_inputs(module, root: Path) -> None:
    binary = root / "bin/aiperf-runner"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x7fELFtest-runner")
    binary.chmod(0o755)
    digest = module.blake3()
    digest.update(module._DISTRIBUTION_ID_DOMAIN)
    digest.update(binary.read_bytes())
    manifest = {
        "cargo_lock_sha256": "sha256:" + "b" * 64,
        "dependency_revisions": {},
        "distribution_id": f"blake3:{digest.hexdigest()}",
        "features": [],
        "schema_version": 2,
        "source_revision": "a" * 40,
    }
    (root / "bin/runner-build.json").write_bytes(module.orjson.dumps(manifest))


def test_companion_and_frontend_versions_move_together() -> None:
    frontend = _toml("pyproject.toml")["project"]
    companion = _toml("packaging/aiperf-runner/pyproject.toml")["project"]

    assert companion["name"] == "aiperf-runner"
    assert companion["version"] == frontend["version"]
    assert companion["requires-python"] == frontend["requires-python"]


def test_companion_has_no_python_runtime_entrypoint() -> None:
    project = _toml("packaging/aiperf-runner/pyproject.toml")
    wheel = project["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert "scripts" not in project["project"]
    assert "entry-points" not in project["project"]
    assert wheel["bypass-selection"] is True
    assert wheel["hooks"]["custom"]["path"] == "hatch_build.py"


def test_companion_hook_and_stager_share_one_strict_registry_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook = _build_hook(monkeypatch)

    assert hook._PROVIDER_ROOT_SPECS == PROVIDER_ROOT_SPECS
    assert hook._PROVIDER_ROOTS_SCHEMA == PROVIDER_ROOTS_SCHEMA
    assert hook._PROVIDER_ROOTS_REGISTRY == PROVIDER_ROOTS_REGISTRY


def test_companion_hook_requires_exact_amd64_evaluator_root_atom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = _build_hook(monkeypatch)
    staged = tmp_path / "evaluator-roots"

    with pytest.raises(RuntimeError, match="requires staged evaluator roots"):
        hook._validate_provider_roots(staged)

    _staged_roots(hook, staged)
    hook._validate_provider_roots(staged)
    (staged / "nemo/proof.txt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="content tree drifted"):
        hook._validate_provider_roots(staged)


def test_companion_hook_rejects_a_symlinked_staged_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = _build_hook(monkeypatch)
    staged = tmp_path / "evaluator-roots"
    _staged_roots(hook, staged)
    relocated = tmp_path / "relocated-nemo"
    (staged / "nemo").rename(relocated)
    (staged / "nemo").symlink_to(relocated, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink|special file"):
        hook._validate_provider_roots(staged)


def test_companion_hook_places_the_complete_root_atom_in_wheel_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _build_hook(monkeypatch)
    _runner_inputs(module, tmp_path)
    roots = tmp_path / module._PROVIDER_ROOTS_SOURCE
    _staged_roots(module, roots)
    monkeypatch.setenv("AIPERF_RUNNER_WHEEL_PLATFORM_TAG", "manylinux_2_28_x86_64")
    build_data = {"extra_metadata": {}, "shared_scripts": {}}
    hook = module.CustomBuildHook()
    hook.root = str(tmp_path)

    hook.initialize("0.11.0", build_data)

    assert build_data["tag"] == "py3-none-manylinux_2_28_x86_64"
    assert build_data["force_include"] == {
        str(roots): module._PROVIDER_ROOTS_WHEEL_TARGET
    }

    monkeypatch.setenv("AIPERF_RUNNER_WHEEL_PLATFORM_TAG", "manylinux_2_28_aarch64")
    with pytest.raises(RuntimeError, match="only by Linux x86_64"):
        hook.initialize("0.11.0", {"extra_metadata": {}, "shared_scripts": {}})


def test_companion_hook_limits_stock_roots_to_linux_amd64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook = _build_hook(monkeypatch)

    assert hook._supports_stock_evaluators("manylinux_2_28_x86_64")
    assert hook._supports_stock_evaluators("manylinux2014_x86_64")
    assert hook._supports_stock_evaluators("musllinux_1_2_x86_64")
    assert hook._supports_stock_evaluators("linux_x86_64")
    assert not hook._supports_stock_evaluators("manylinux_2_28_aarch64")
    assert not hook._supports_stock_evaluators("macosx_14_0_x86_64")
    assert not hook._supports_stock_evaluators("notlinux_x86_64")


def test_companion_hook_hashes_large_payloads_with_bounded_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = _build_hook(monkeypatch)
    content = b"z" * (2 * 1024 * 1024 + 31)
    payload = tmp_path / "large-payload"
    payload.write_bytes(content)

    assert hook._file_sha256(payload) == hashlib.sha256(content).hexdigest()


def test_container_and_release_require_verified_companion_inputs() -> None:
    dockerfile = (_ROOT / "Dockerfile").read_text()
    nightly = (_ROOT / ".github/workflows/nightly.yml").read_text()

    assert "/dist/aiperf_runner-*.whl" in dockerfile
    assert "verify_runner_companion.py" in dockerfile
    assert '--profile "${AIPERF_RUNNER_PROFILE}"' in dockerfile
    assert "AIPERF_RUNNER_PROFILE=offline" in dockerfile
    assert "runner-build.json" in nightly
    assert "refuses to synthesize Cargo metadata or substitute" in nightly
    assert (
        "aiperf/runner-inputs/${RESOLVED_SHA}/linux-${{ matrix.arch }}/${PROFILE}"
        in nightly
    )
    assert "for PROFILE in online offline" in nightly
    assert 'verify_runner_companion.py --profile "${PROFILE}"' in nightly
    assert "--build-arg AIPERF_RUNNER_PROFILE=offline" in nightly


def test_platform_ci_builds_and_executes_both_native_profiles() -> None:
    workflow = (_ROOT / ".github/workflows/native-runner.yml").read_text()

    assert "prod-aiperf-builder-amd-v1" in workflow
    assert "prod-aiperf-builder-arm-v1" in workflow
    assert "cargo test --locked --workspace" in workflow
    assert "--test stdio_e2e" in workflow
    assert "--test offline_stdio" in workflow
    assert "--test offline_scheduled_stdio" in workflow
    assert "cargo build --locked --release -p aiperf-runner" in workflow
    assert "--features dynamo-offline" in workflow
    assert "tools/runner_release_input.py create" in workflow
    assert "tools/runner_release_input.py verify" in workflow
    assert "dynamo-aiperf-native=${DYNAMO_REVISION}" in workflow
    assert "aiperf/runner-inputs/${GITHUB_SHA}/linux-${{ matrix.arch }}" in workflow
